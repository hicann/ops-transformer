#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""aclgraph 编译目标模块 (quant_flash_mla_with_kvcache)。

QuantMlaAclGraph 是 torch.nn.Module，__init__ 在 capture 之外完成全部预处理
(golden 全局变量注入、generate_data、layout 转换、metadata 构建)，
forward() 只调 quant_flash_mla_with_kvcache 主算子，可被 npugraph_ex 捕获。
"""

import logging
import os
import sys

import torch

# 复用 qmla_with_kvcache_golden 的 generate_data / layout 转换 / 全局变量
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)
# golden 模块依赖 pytest/common/ 下的 result_compare_method
_COMMON_DIR = os.path.join(_ASSETS_DIR, "..", "..", "pytest", "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

logger = logging.getLogger(__name__)


def _load_npu_modules():
    """懒加载 torch_npu 与 golden 模块。

    延迟到 worker (fork 之后) 再 import torch_npu, 避免主进程 fork 前 import
    torch_npu 导致 fork 子进程 re-init NPU 报错。
    """
    import torch_npu
    import qmla_with_kvcache_golden as golden_mod

    return torch_npu, golden_mod


_KV_LAYOUT_REVERSE = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


def _apply_golden_globals(params, golden_mod):
    """把 case 参数注入 golden 模块全局变量。

    generate_data / convert_q_bnsd_to_layout / bnsd_to_k_cache 等函数
    读取 golden_mod 的全局变量 (B/N_q/N_kv/D/ENABLE_PA/INPUT_LAYOUT/...)，
    必须在调用前全部注入。
    """
    for k, v in params.items():
        setattr(golden_mod, k, v)


class QuantMlaAclGraph(torch.nn.Module):
    """aclgraph 编译目标: forward 只调 quant_flash_mla_with_kvcache 主算子。

    __init__ (capture 之外) 完成:
      1. 从函数参数取 final-layout fp8 q/k_cache + descale + block_table + cache_seqlens
      2. 注入 golden 全局变量 (B/N_q/.../ENABLE_PA/INPUT_LAYOUT/...)
      3. generate_data 生成 BNSD FP8 数据 → 转目标 layout + PA cache
      4. cu_seqlens_q/seqused_q 直接用入参 tensor (inputs.py 已写入真实值)
      5. quant_flash_mla_with_kvcache_metadata 构建 (capture 之前, metadata 是 int32 tensor)
      6. 所有运行时入参存为 self. 属性, forward 直接读

    入参对齐 10 位置 tensor (与 CSV tensor_view_shapes 一致):
      0 q, 1 k_cache, 2 q_descale, 3 k_descale, 4 block_table,
      5 cache_seqlens, 6 cu_seqlens_q, 7 seqused_q, 8 attn_mask, 9 metadata。

    forward() (capture 之内) 只调:
      torch.ops.cann_ops_transformer.quant_flash_mla(...)
    """

    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        quant_mode: int,
        *,
        cu_seqlens_q=None,
        seqused_q=None,
        attn_mask=None,
        metadata=None,
        softmax_scale=1.0,
        mask_mode=0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_v=512,
        layout_q="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
        return_softmax_lse=False,
        **kwargs,
    ):
        super().__init__()
        torch_npu_mod, golden_mod = _load_npu_modules()
        torch_npu_mod.npu.set_device(int(kwargs.get("device_id", 0)))

        # ---- 0. 读取 sidecar attributes ----
        B = kwargs.get("B", 1)
        N_q = kwargs.get("N_q", 6)
        N_kv = kwargs.get("N_kv", 1)
        D = kwargs.get("D", 512)
        D_V = kwargs.get("D_V", 512)
        D_rope = kwargs.get("D_rope", 64)
        block_size = kwargs.get("block_size", 128)
        seqused_q_vals = kwargs.get("seqused_q_values", [1])
        cache_seqlens_vals = kwargs.get("cache_seqlens_values", [1024])
        kv_cache_layout = kwargs.get("kv_cache_layout") or _KV_LAYOUT_REVERSE.get(
            layout_kv, "BnNBsD"
        )
        input_layout = kwargs.get("input_layout", layout_q)
        output_layout = kwargs.get("output_layout", layout_out)
        p_scale_val = float(kwargs.get("p_scale", 1.0))
        graph_path = kwargs.get("graph_path", 0)

        # cu_seqlens_q / seqused_q: 从 tensor 读回 list (inputs.py 已 in-place 写入)
        def _tolist_t(t, default=None):
            if t is None:
                return default
            return t.detach().cpu().tolist() if t.numel() > 0 else default

        cu_seqlens_q_list = _tolist_t(cu_seqlens_q, [0])
        seqused_q_list = _tolist_t(seqused_q, list(seqused_q_vals))

        # ---- 1. 注入 golden 全局变量 ----
        _apply_golden_globals(
            {
                "B": B,
                "N_q": N_q,
                "N_kv": N_kv,
                "D": D,
                "D_V": D_V,
                "D_rope": D_rope,
                "QMLA_QK_HEAD_DIM": D + D_rope,
                "QMLA_V_HEAD_DIM": D_V,
                "QMLA_QUANT_MODE": quant_mode,
                "SEQUSED_Q": list(seqused_q_vals),
                "CACHE_SEQLENS": list(cache_seqlens_vals),
                "ENABLE_PA": True,
                "ENABLE_LSE": return_softmax_lse,
                "GOLDEN_MODE": True,
                "BLOCK_SIZE": block_size,
                "MASK_MODE": mask_mode,
                "INPUT_LAYOUT": input_layout,
                "OUTPUT_LAYOUT": output_layout,
                "KV_CACHE_LAYOUT": kv_cache_layout,
                "P_SCALE": p_scale_val,
                "GRAPH_PATH": graph_path,
                "NUM_BLOCKS": kwargs.get("num_blocks", 0),
                "SCALE_VALUE": softmax_scale,
                "MAX_SEQLEN_Q": max_seqlen_q,
                "MAX_SEQLEN_KV": max_seqlen_kv,
            },
            golden_mod,
        )

        # Data ranges
        if kwargs.get("q_data_range"):
            golden_mod.Q_DATA_RANGE = tuple(kwargs["q_data_range"])
        if kwargs.get("k_data_range"):
            golden_mod.K_DATA_RANGE = tuple(kwargs["k_data_range"])
        if kwargs.get("v_data_range"):
            golden_mod.V_DATA_RANGE = tuple(kwargs["v_data_range"])

        # Seeds
        for s in (
            "SEED_Q",
            "SEED_K",
            "SEED_V",
            "SEED_QR",
            "SEED_KR",
            "SEED_BLOCK_TABLE",
        ):
            key = s.lower()
            if key in kwargs:
                setattr(golden_mod, s, kwargs[key])

        # ---- 2. generate_data (BNSD FP8, CPU) ----
        (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale_t, qr_bf16, kr_bf16) = (
            golden_mod.generate_data(device="cpu")
        )

        logger.info(
            "[GRAPH] generate_data: q=%s, k=%s, deq_q=%s, deq_k=%s",
            tuple(q_fp8.shape),
            tuple(k_fp8.shape),
            tuple(deq_q.shape),
            tuple(deq_k.shape),
        )

        # ---- 3. 转目标 layout ----
        q_final = golden_mod.convert_q_bnsd_to_layout(
            q_fp8, seqused_q_vals, input_layout
        )

        # block_table
        block_table_t = golden_mod.create_block_table(
            cache_seqlens_vals, block_size, num_blocks=golden_mod.NUM_BLOCKS
        )

        # K cache
        k_pa = golden_mod.bnsd_to_k_cache(
            k_fp8,
            cache_seqlens_vals,
            block_size,
            block_table_t,
            num_blocks=golden_mod.NUM_BLOCKS,
            kv_layout=kv_cache_layout,
        )

        # descale
        deq_q_final = golden_mod.convert_scale_to_layout(deq_q, seqused_q_vals, "deq_q")
        deq_k_final = deq_k.float().reshape(-1)[:1].contiguous()

        # ---- 4. 移到 NPU ----
        self.q = q_final.npu()
        self.k_cache = k_pa.npu()
        self.q_descale = deq_q_final.npu()
        self.k_descale = deq_k_final.npu()
        self.block_table = block_table_t.npu()
        self.cache_seqlens = torch.tensor(cache_seqlens_vals, dtype=torch.int32).npu()

        # cu_seqlens_q (TND only)
        is_tnd = layout_q == "TND"
        if is_tnd:
            self.cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32).npu()
        else:
            self.cu_seqlens_q = None

        # seqused_q
        self.seqused_q = torch.tensor(seqused_q_vals, dtype=torch.int32).npu()

        # attn_mask (causal mask_mode=3)
        if mask_mode == 3:
            mask_shape = (2048, 2048)
            causal_mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.int8), diagonal=1
            ).npu()
            self.attn_mask = causal_mask
        else:
            self.attn_mask = None

        torch.npu.synchronize()

        # ---- 5. metadata 构建 (capture 之前) ----
        head_dim_qk = D + D_rope

        # max_seqlen auto-derive
        if max_seqlen_q is None or max_seqlen_q <= 0:
            max_seqlen_q = max(seqused_q_vals) if seqused_q_vals else 8
        if max_seqlen_kv is None or max_seqlen_kv <= 0:
            max_seqlen_kv = max(cache_seqlens_vals) if cache_seqlens_vals else 1024

        logger.info("[GRAPH] 构建 metadata (quant_flash_mla_with_kvcache_metadata)")
        self.metadata = torch.ops.cann_ops_transformer.quant_flash_mla_metadata(
            cache_seqlens=self.cache_seqlens,
            num_heads_q=N_q,
            num_heads_kv=N_kv,
            quant_mode=quant_mode,
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_q=self.seqused_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            mask_mode=mask_mode,
            layout_q=layout_q,
        )
        logger.info("[GRAPH] metadata shape=%s", tuple(self.metadata.shape))

        # ---- 6. 存 forward 需要的属性 ----
        self.quant_mode = int(quant_mode)
        self.softmax_scale = softmax_scale
        self.mask_mode = int(mask_mode)
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_kv = int(max_seqlen_kv)
        self.head_dim_v = int(head_dim_v)
        self.return_softmax_lse = bool(return_softmax_lse)

        logger.info(
            "[GRAPH] __init__ done: q=%s, k_cache=%s, layout_q=%s, layout_kv=%s, "
            "metadata=%s",
            self.q.shape,
            self.k_cache.shape,
            self.layout_q,
            self.layout_kv,
            self.metadata.shape,
        )

    def forward(self):
        """只调 quant_flash_mla_with_kvcache 主算子, 可被 npugraph_ex 捕获。

        所有入参来自 self. 属性 (__init__ 预处理结果), forward 无参数。
        """
        attn_out, lse_out = torch.ops.cann_ops_transformer.quant_flash_mla(
            self.q,
            self.k_cache,
            self.q_descale,
            self.k_descale,
            self.block_table,
            self.cache_seqlens,
            self.quant_mode,
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_q=self.seqused_q,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            head_dim_v=self.head_dim_v,
            layout_q=self.layout_q,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            return_softmax_lse=self.return_softmax_lse,
        )

        if not self.return_softmax_lse:
            lse_out = None
        return attn_out, lse_out
