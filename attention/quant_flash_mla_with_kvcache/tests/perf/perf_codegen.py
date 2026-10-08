#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""生成单 case 自包含性能脚本(字符串), 由 run_perf.py 调用, 经 msprof 子进程执行。

数据构造对齐 tests/pytest/common/qmla_with_kvcache_golden.py 的 NPU 路径:
  - Q: fp16 BNSD -> per-token-head 量化 -> TND (T, N_q, 576), descale (T, N_q)
  - K: fp16 BNSD -> per-tensor 量化 -> PA cache (BnNBsD / BnBsH / NZ), descale 标量
  - V 由算子内部从 K 的前 512 维复用, 无需单独传入
两段式调用: metadata(循环外一次) + 主算子热循环。
注意: 新接口 block_table 为必填参数, 因此非 PA 用例也统一按 PA 形式构造
(block table 顺序映射, kernel 数据量与原语义等价)。
"""

# 固定量纲(MLA): nope 512 + rope 64 = 576
D_NOPE = 512
D_ROPE = 64
D_V = 512
BLOCK_SIZE = 128

_TEMPLATE = r"""import math
import traceback

import torch
import torch_npu

try:
    from cann_ops_transformer.ops import (
        quant_flash_mla_with_kvcache,
        quant_flash_mla_with_kvcache_metadata,
    )
except ImportError as e:
    print(f'[ERROR] import failed: {e} (请先 source CANN set_env.sh)')
    raise SystemExit(1)

print('CASE: @NAME@  B=@B@ N_q=@NQ@ N_kv=@NKV@ kv=@KV_LAYOUT@ mask=@MASK_MODE@ lse=@ENABLE_LSE@ runs=@RUNS@', flush=True)

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

D_NOPE = @D_NOPE@
D_ROPE = @D_ROPE@
DQK = D_NOPE + D_ROPE
HEAD_DIM_V = @D_V@
B = @B@
N_Q = @NQ@
N_KV = @NKV@
SEQUSED_Q = @SEQUSED_Q@
CACHE_SEQLENS = @CACHE_SEQLENS@
MAX_SQ = max(SEQUSED_Q)
MAX_SKV = max(CACHE_SEQLENS)
BLOCK_SIZE = @BLOCK_SIZE@
MASK_MODE = @MASK_MODE@
ENABLE_LSE = @ENABLE_LSE@
KV_CACHE_LAYOUT = '@KV_LAYOUT@'
NUM_BLOCKS = @NUM_BLOCKS@
LAYOUT_Q = 'TND'
SOFTMAX_SCALE = 1.0 / math.sqrt(D_NOPE)

try:
    torch.manual_seed(20260914)
    T = sum(SEQUSED_Q)

    # ---- Q: fp16 BNSD -> per-token-head 量化 ----
    q_fp16 = torch.empty(B, N_Q, MAX_SQ, DQK, dtype=torch.float16).uniform_(-1.0, 1.0)
    q_scale = FP8_MAX / q_fp16.float().abs().amax(dim=3, keepdim=True).clamp_min(1e-8)  # (B,N,S,1)
    q_fp8 = (q_fp16.float() * q_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    deq_q_bnsd = (1.0 / q_scale).squeeze(-1)  # (B,N,S)

    # BNSD -> TND: 按 batch 实际长度拼接
    q_tnd = torch.cat(
        [q_fp8[b, :, :SEQUSED_Q[b], :].permute(1, 0, 2) for b in range(B)], dim=0
    ).contiguous()  # (T, N_Q, DQK)
    deq_q = torch.cat(
        [deq_q_bnsd[b, :, :SEQUSED_Q[b]].transpose(0, 1) for b in range(B)], dim=0
    ).float().contiguous()  # (T, N_Q)

    # ---- K: fp16 BNSD -> per-tensor 量化 ----
    k_fp16 = torch.empty(B, N_KV, MAX_SKV, DQK, dtype=torch.float16).uniform_(-5.0, 5.0)
    k_scale = FP8_MAX / max(k_fp16.float().abs().max().item(), 1e-8)
    k_fp8 = (k_fp16.float() * k_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).contiguous()
    deq_k = torch.tensor([1.0 / k_scale], dtype=torch.float32)

    # ---- block table (num_blocks 控制物理块复用, 语义同 golden.create_block_table) ----
    import numpy as np
    blk_per_b = [math.ceil(int(l) / BLOCK_SIZE) for l in CACHE_SEQLENS]
    total_blocks = sum(blk_per_b)
    cache_blocks = NUM_BLOCKS if NUM_BLOCKS != 0 else total_blocks
    rng = np.random.default_rng(1234)
    if NUM_BLOCKS != 0 and NUM_BLOCKS < total_blocks:
        idx = rng.integers(0, NUM_BLOCKS, size=total_blocks, dtype=np.int32)
    elif NUM_BLOCKS != 0:
        idx = rng.permutation(np.arange(NUM_BLOCKS, dtype=np.int32))
    else:
        idx = np.arange(total_blocks, dtype=np.int32)
    block_table_np = np.full((B, max(blk_per_b)), -1, dtype=np.int32)
    _i = 0
    for b in range(B):
        block_table_np[b, :blk_per_b[b]] = idx[_i:_i + blk_per_b[b]]
        _i += blk_per_b[b]

    # ---- K BNSD -> PA cache (纯数据 block; V 由算子取 K 前 D_NOPE 维) ----
    k_cache = torch.zeros(cache_blocks, N_KV, BLOCK_SIZE, DQK, dtype=FP8)
    for b in range(B):
        for j in range(blk_per_b[b]):
            bid = int(block_table_np[b, j])
            s0 = j * BLOCK_SIZE
            s1 = min(s0 + BLOCK_SIZE, CACHE_SEQLENS[b])
            if s1 <= s0:
                continue
            k_cache[bid, :, :s1 - s0, :] = k_fp8[b, :, s0:s1, :]

    # ---- PA 布局转换 (语义同 golden._pa_layout_transform) ----
    if KV_CACHE_LAYOUT == 'BnBsH':
        bn, n, bs, d = k_cache.shape
        k_cache = k_cache.transpose(1, 2).reshape(bn, bs, n * d).contiguous()
    elif KV_CACHE_LAYOUT == 'NZ':
        bn, n, bs, d = k_cache.shape
        d0 = 32 // k_cache.element_size()
        d1 = d // d0
        k_cache = k_cache.reshape(bn, n, bs, d1, d0).permute(0, 1, 3, 2, 4).contiguous()

    # ---- 上 NPU ----
    q_npu = q_tnd.npu().contiguous()
    k_cache_npu = k_cache.npu().contiguous()
    deq_q_npu = deq_q.npu().contiguous()
    deq_k_npu = deq_k.npu().contiguous()
    p_scale = torch.tensor([1.0], dtype=torch.float32).npu().contiguous()
    block_table = torch.as_tensor(block_table_np, dtype=torch.int32).npu().contiguous()
    cache_seqlens_t = torch.tensor(CACHE_SEQLENS, dtype=torch.int32).npu().contiguous()
    cu_q = [0]
    _acc = 0
    for _s in SEQUSED_Q:
        _acc += int(_s)
        cu_q.append(_acc)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32).npu().contiguous()
    seqused_q_t = torch.tensor(SEQUSED_Q, dtype=torch.int32).npu().contiguous()
    if MASK_MODE == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None
    torch.npu.synchronize()

    # ---- metadata (循环外一次; AI_CPU 算子, 不计入 kernel 时长) ----
    metadata = quant_flash_mla_with_kvcache_metadata(
        cache_seqlens=cache_seqlens_t,
        num_heads_q=N_Q,
        num_heads_kv=N_KV,
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q_t,
        max_seqlen_q=MAX_SQ,
        max_seqlen_kv=MAX_SKV,
        head_dim_qk=DQK,
        head_dim_v=HEAD_DIM_V,
        mask_mode=MASK_MODE,
        layout_q=LAYOUT_Q,
    )

    KV_LAYOUT_MAP = {'BnNBsD': 'PA_BNBD', 'BnBsH': 'PA_BBND', 'NZ': 'PA_NZ'}
    layout_kv = KV_LAYOUT_MAP.get(KV_CACHE_LAYOUT, 'PA_BNBD')

    print(f'  -> tensors ready, running @RUNS@ iterations (hot)...', flush=True)
    torch.npu.synchronize()
    for _run_i in range(@RUNS@):
        torch.npu.synchronize()
        atten_out, lse_out = quant_flash_mla_with_kvcache(
            q=q_npu,
            k_cache=k_cache_npu,
            q_descale=deq_q_npu,
            k_descale=deq_k_npu,
            block_table=block_table,
            cache_seqlens=cache_seqlens_t,
            quant_mode=1,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q_t,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=SOFTMAX_SCALE,
            mask_mode=MASK_MODE,
            max_seqlen_q=MAX_SQ,
            max_seqlen_kv=MAX_SKV,
            head_dim_v=HEAD_DIM_V,
            layout_q=LAYOUT_Q,
            layout_kv=layout_kv,
            layout_out=LAYOUT_Q,
            return_softmax_lse=ENABLE_LSE,
        )
    torch.npu.synchronize()
    print('  -> DONE', flush=True)
except Exception:
    traceback.print_exc()
    print('  -> FAILED', flush=True)
finally:
    import gc
    gc.collect()
    torch.npu.empty_cache()
print('done')
"""


def build_script(name: str, case: dict, runs: int) -> str:
    """生成单 case 性能脚本。

    case 字段(schema 与 perf_cases.py 一致):
      B, N_q, N_kv, seqused_q, cache_seqlens,
      enable_pa(仅标注, 统一按 PA 构造), kv_cache_layout,
      sparse_mode(即 mask_mode), enable_lse, num_blocks
    """
    subs = {
        "@NAME@": str(name),
        "@RUNS@": str(int(runs)),
        "@D_NOPE@": str(D_NOPE),
        "@D_ROPE@": str(D_ROPE),
        "@D_V@": str(D_V),
        "@B@": str(int(case["B"])),
        "@NQ@": str(int(case["N_q"])),
        "@NKV@": str(int(case["N_kv"])),
        "@SEQUSED_Q@": repr([int(x) for x in case["seqused_q"]]),
        "@CACHE_SEQLENS@": repr([int(x) for x in case["cache_seqlens"]]),
        "@BLOCK_SIZE@": str(BLOCK_SIZE),
        "@MASK_MODE@": str(int(case.get("sparse_mode", 0))),
        "@ENABLE_LSE@": str(bool(case.get("enable_lse", False))),
        "@KV_LAYOUT@": str(case.get("kv_cache_layout", "BnNBsD")),
        "@NUM_BLOCKS@": str(int(case.get("num_blocks", 0))),
    }
    script = _TEMPLATE
    for token, value in subs.items():
        script = script.replace(token, value)
    return script
