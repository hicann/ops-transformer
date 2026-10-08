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
"""
FIA FullQuant MLA Golden

功能：生成 BNSD 数据 → nope(512)+rope(64) 合并为 576 维 → FP8 per-token-head (Q) / per-tensor (K) 量化
      → NPU golden (可选 CPU) → layout 转换 → NPU 调用 → 精度对比
支持：PA / 非 PA 场景，MLA (DeepSeek 风格: D_nope=512 + D_rope=64, qkHeadDim=576, dV=512)
量化：Q per-token-head (整 576 维), K per-tensor (整 576 维)
MLA 特性: bmm1 = Q@K^T (576 维合并, 统一 descale), bmm2 = P@V (V = K 的 nope 前 512 维)
          softmax_scale = 1/sqrt(D) (仅 nope 维度 D=512, 不含 rope)
"""

import argparse
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng

if __package__:
    from . import result_compare_method
else:
    import result_compare_method

try:
    from cann_ops_transformer_qmla.ops.attention.quant_flash_mla_with_kvcache import (
        quant_flash_mla_with_kvcache,
        quant_flash_mla_with_kvcache_metadata,
    )
except ImportError:
    try:
        from cann_ops_transformer.ops import (
            quant_flash_mla_with_kvcache,
            quant_flash_mla_with_kvcache_metadata,
        )
    except ImportError:
        quant_flash_mla_with_kvcache = None
        quant_flash_mla_with_kvcache_metadata = None

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 7=aclgraph, 8=aclgraph+静态kernel
GRAPH_PATH = 0
# QMLA_PROF=1: 仅对 fa 测量循环 (warmup 之后的 runs) 开启 torch_npu profiler,
# golden/数据准备不进 profiling, 避免逐分段模拟小算子污染 stream 与结果
QMLA_PROF = os.environ.get("QMLA_PROF", "0").strip().lower() in ("1", "true")
DEVICE_ID = 0
OP_WARMUP = 0
OP_RUNS = 1
OP_FLUSH_L2 = False

B = 1
N_q = 96
N_kv = 1

# MLA dims: D_nope + D_rope = qkHeadDim
D = 512  # nope head dim (Q/K)
D_V = 512  # V head dim
D_rope = 64  # rope head dim
# 新接口 quant_flash_mla_with_kvcache 常量
QMLA_QUANT_MODE = 1  # 1=FP8_E4M3（mla 新接口）
QMLA_QK_HEAD_DIM = D + D_rope  # 576 = nope(512) + rope(64)，q/k_cache 沿末维拼接
QMLA_V_HEAD_DIM = D_V  # 512

SEQUSED_Q = [1] * B
CACHE_SEQLENS = [102408] * B
MAX_SEQLEN_Q = -1  # -1 means auto-derive from SEQUSED_Q
MAX_SEQLEN_KV = -1  # -1 means auto-derive from CACHE_SEQLENS
# Layout 选择 (TND: query 排布 (T, N_q, D); deq_q 用 TN (T, N_q) 第0维与 query 一致)
INPUT_LAYOUT = "TND"
OUTPUT_LAYOUT = INPUT_LAYOUT

# PA KV Cache Layout：BnBsND、BnNBsD、NZ
KV_CACHE_LAYOUT = "NZ"  # 新接口仅支持 PA_BNBD

# Data Range (lo, hi)
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-5.0, 5.0)
V_DATA_RANGE = (-5.0, 5.0)
# rope 部分数据范围 (与 nope 拼成 576 维 q/k 后统一量化)
ROPE_DATA_RANGE = (-0.3, 0.3)

ENABLE_PA = True
ENABLE_LSE = False
GOLDEN_MODE = True
BLOCK_SIZE = 128
MASK_MODE = 3
SCALE_VALUE = None
IS_CONTIGUOUS = True

# Q scale layout 由 INPUT_LAYOUT 自动推导, 不再暴露独立的 Q_SCALE_LAYOUT 字段
_Q_SCALE_LAYOUT_MAP = {
    "BNSD": "BNS",
    "BSND": "BSN",
    "TND": "TN",
    "NTD_TND": "NT",
}

# KV_CACHE_LAYOUT → op layout_kv 映射
_KV_LAYOUT_MAP = {
    "BnNBsD": "PA_BNBD",
    "BnBsND": "PA_BBND",
    "NZ": "PA_NZ",
}

# Seed
SEED_Q = 54
SEED_K = 3
SEED_V = 20
SEED_QR = 8
SEED_KR = 9
SEED_BLOCK_TABLE = 1234

FP8_DTYPE = torch.float8_e4m3fn
OUTPUT_DETYPE = torch.bfloat16
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 64
KV_BLOCK_SIZE = 128

# 物理 block 数量，0 表示使用默认值（等于 total_blocks）
NUM_BLOCKS = 0


# ==============================================================================
# 量化 scale 计算
# MLA 量化:
#   Q/K: per-token-head, scale shape (B, N, S, 1)
#   V:   per-tensor,     scale shape (1,)  (全 tensor 一个 scale)
# ==============================================================================
def get_fp8_per_token_head_quant_scale(tensor):
    """per-token-head quant scale: shape (B, N, S, 1)"""
    tensor = tensor.contiguous()
    B, N, S, _ = tensor.shape
    fp8_e4m3_max = 448.0
    row_max = torch.abs(tensor).max(dim=3, keepdim=True).values
    row_max = torch.max(row_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / row_max
    return scale.view(B, N, S, 1).float().contiguous()


def get_fp8_per_tensor_quant_scale(tensor):
    """per-tensor quant scale: shape (1,)"""
    tensor = tensor.contiguous()
    fp8_e4m3_max = 448.0
    tensor_max = torch.abs(tensor).max()
    tensor_max = torch.max(tensor_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / tensor_max
    return scale.reshape(1).float().contiguous()


def quant_fp16_to_fp8(tensor, scale):
    """将 fp16/bf16 数据量化为 fp8_e4m3"""
    tensor = tensor.contiguous()
    scale = scale.contiguous()
    result = tensor.float() * scale
    result = torch.clamp(result, -448.0, 448.0)
    return result.to(FP8_DTYPE).contiguous()


# ==============================================================================
# Block table / PA cache 工具
# ==============================================================================
def create_block_table(cache_seqlens, block_size, seed=SEED_BLOCK_TABLE, num_blocks=0):
    """创建 block table，num_blocks 控制物理块复用"""
    block_num_per_batch = [
        math.ceil(int(seq_len) / block_size) for seq_len in cache_seqlens
    ]
    total_blocks = sum(block_num_per_batch)
    max_blocks = max(block_num_per_batch)

    if num_blocks < total_blocks and num_blocks != 0:
        block_idx_list = np.random.default_rng(seed).integers(
            0, num_blocks, size=total_blocks, dtype=np.int32
        )
    elif num_blocks > total_blocks and num_blocks != 0:
        block_idx_list = np.random.default_rng(seed).permutation(
            np.arange(num_blocks, dtype=np.int32)
        )
    else:
        block_idx_list = np.random.default_rng(seed).permutation(
            np.arange(total_blocks, dtype=np.int32)
        )

    block_table = np.full((len(cache_seqlens), max_blocks), -1, dtype=np.int32)
    idx = 0
    for b_index, block_num in enumerate(block_num_per_batch):
        block_table[b_index, :block_num] = block_idx_list[idx : idx + block_num]
        idx += block_num
    return block_table


def _pa_layout_transform(out_cache, kv_layout, d_dim):
    """将 BnNBsD (Bn, N, Bs, D) 形式的 PA cache 转换为目标 kv_layout

    MLA PA_NZ: (Bn, N, D/16, Bs, 32/sizeof(qdtype))
      - int8/fp8: d0 = 32, d1 = D/16
      - bf16:     d0 = 16, d1 = D/16
    """
    if kv_layout == "BnNBsD":
        return out_cache.contiguous()
    elif kv_layout == "BnBsND":
        bn, n, bs, d = out_cache.shape
        return out_cache.transpose(1, 2).reshape(bn, bs, n, d).contiguous()
    elif kv_layout == "NZ":
        bn, n, bs, d = out_cache.shape
        d0 = 32 // out_cache.element_size()
        d1 = d_dim // d0
        reshaped = out_cache.reshape(bn, n, bs, d1, d0)
        return reshaped.permute(0, 1, 3, 2, 4).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def bnsd_to_k_cache(
    k_fp8_bnsd, seq_lens, block_size, block_table, num_blocks=0, kv_layout="BnNBsD"
):
    """BNSD to PA K cache - 纯数据 block (MLA K per-tensor, scale 不与 key 共享内存)"""
    k_fp8_bnsd = k_fp8_bnsd.contiguous()
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    out_cache = torch.zeros(
        (cache_blocks, N_dim, block_size, D_dim),
        dtype=FP8_DTYPE,
        device=k_fp8_bnsd.device,
    ).contiguous()

    for b in range(B_dim):
        bid_table = block_table[b]
        for blk_idx in range(block_num_per_seq[b]):
            blockid = int(bid_table[blk_idx])
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_lens[b])
            valid = end_s - start_s
            if valid <= 0:
                continue
            out_cache[blockid, :, :valid, :] = k_fp8_bnsd[
                b, :, start_s:end_s, :
            ].contiguous()

    return _pa_layout_transform(out_cache, kv_layout, D_dim)


def bnsd_to_v_cache(
    tensor_bnsd, seq_lens, block_size, block_table, num_blocks=0, kv_layout="BnNBsD"
):
    """BNSD to V cache - 纯数据 block (MLA V per-tensor, scale 不共享内存)"""
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    out_cache = torch.zeros(
        (cache_blocks, heads, block_size, dim), dtype=FP8_DTYPE, device=device
    ).contiguous()

    for b in range(batch):
        for blk_idx in range(block_num_per_batch[b]):
            block_id = int(block_table[b, blk_idx].item())
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[block_id, :, :valid_len, :] = tensor_bnsd[
                b, :, block_offset : block_offset + valid_len, :
            ].contiguous()

    return _pa_layout_transform(out_cache, kv_layout, dim)


def _bnsd_to_pa_bf16(
    tensor_bnsd, seq_lens, block_size, block_table, num_blocks=0, kv_layout="BnNBsD"
):
    """BNSD bf16 (rope) → PA cache (无 scale rows, 纯数据 block 分片)

    输出: (cache_blocks, N, block_size, D_rope) bf16, 并按 kv_layout 转换
    """
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    out_cache = torch.zeros(
        (cache_blocks, heads, block_size, dim), dtype=tensor_bnsd.dtype, device=device
    ).contiguous()

    for b in range(batch):
        for blk_idx in range(block_num_per_batch[b]):
            block_id = int(block_table[b, blk_idx].item())
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[block_id, :, :valid_len, :] = tensor_bnsd[
                b, :, block_offset : block_offset + valid_len, :
            ].contiguous()

    return _pa_layout_transform(out_cache, kv_layout, dim)


# ==============================================================================
# 数据生成
# ==============================================================================
def generate_data(device="npu"):
    """生成 BNSD FP16 Q/K (nope+rope 合并, head_dim_qk=576) 并做 FP8 量化

    MLA 新接口 quant_flash_mla_with_kvcache: q/k_cache 将 nope(512) 与 rope(64)
    沿末维拼接为 576, 统一量化:
      - Q: per-token-head 动态量化 (作用于整 576 维)
      - K: per-tensor 静态量化 (作用于整 576 维)
      - V = K 的 nope 前 512 维 (head_dim_v=512), 共享 K 的 per-tensor scale
    """
    device = torch.device(device)
    logger.info("[Data] generation/quantization device=%s", device)
    max_sq = max(SEQUSED_Q)
    max_skv = max(CACHE_SEQLENS) if max(CACHE_SEQLENS) > 0 else 1
    logger.info("[INFO] max_sq=%d, max_skv=%d", max_sq, max_skv)

    def _generate_one(seed, data_range, shape, amp_shape):
        """用 base * amp 结构生成数据，最后线性映射到精确范围 [data_range[0], data_range[1]]"""
        if device.type == "npu":
            torch.npu.manual_seed(seed)
            amp_hi = max(abs(data_range[0]), abs(data_range[1]))
            amp_lo = max(amp_hi * 0.01, 1e-8)
            log_amps = torch.empty(amp_shape, dtype=torch.float32, device=device)
            log_amps.uniform_(math.log10(amp_lo), math.log10(amp_hi))
            token_amps = torch.pow(10.0, log_amps)
            data = torch.empty(shape, dtype=torch.float32, device=device)
            flat = data.view(-1)
            for start in range(0, flat.numel(), 4 * 1024 * 1024):
                flat[start : start + 4 * 1024 * 1024].uniform_(-1.0, 1.0)
            lo, hi = map(float, data_range)
            data.mul_(token_amps).add_(amp_hi).div_(2 * amp_hi).mul_(hi - lo).add_(lo)
            return data.to(torch.float16)
        np.random.seed(seed)
        amp_hi = max(abs(data_range[0]), abs(data_range[1]))
        amp_lo = max(amp_hi * 0.01, 1e-8)
        token_amps = np.power(
            10.0, np.random.uniform(np.log10(amp_lo), np.log10(amp_hi), size=amp_shape)
        ).astype(np.float32)
        base = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
        raw = base * token_amps
        normed = (raw + amp_hi) / (2.0 * amp_hi)
        lo, hi = float(data_range[0]), float(data_range[1])
        data = lo + normed * (hi - lo)
        return torch.from_numpy(data.astype(np.float16))

    qk_head_dim = D + D_rope  # 576 = nope(512) + rope(64)

    # nope 部分 (D=512)
    q_nope_fp16 = _generate_one(
        SEED_Q, Q_DATA_RANGE, (B, N_q, max_sq, D), (B, N_q, max_sq, 1)
    )
    k_nope_fp16 = _generate_one(
        SEED_K, K_DATA_RANGE, (B, N_kv, max_skv, D), (B, N_kv, max_skv, 1)
    )

    # rope 部分 (D_rope=64)
    q_rope_fp16 = _generate_one(
        SEED_QR, ROPE_DATA_RANGE, (B, N_q, max_sq, D_rope), (B, N_q, max_sq, 1)
    )
    k_rope_fp16 = _generate_one(
        SEED_KR, ROPE_DATA_RANGE, (B, N_kv, max_skv, D_rope), (B, N_kv, max_skv, 1)
    )

    # 合并 nope + rope, 得到统一的 576 维 Q / K (KV 共享, V = K 的 nope 部分)
    q_fp16 = torch.cat([q_nope_fp16, q_rope_fp16], dim=-1).contiguous()
    k_fp16 = torch.cat([k_nope_fp16, k_rope_fp16], dim=-1).contiguous()

    # 量化: Q per-token-head (整 576 维), K per-tensor (整 576 维)
    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_tensor_quant_scale(k_fp16)

    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    # MLA: V 与 K 共享 per-tensor scale
    dequant_scale_v = dequant_scale_k.contiguous()

    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    # MLA: V 使用 K 的 nope 前 D=512 维 (head_dim_v=512)
    v_fp8 = k_fp8[:, :, :, :D].contiguous()

    if max(CACHE_SEQLENS) == 0:
        real_skv = max(CACHE_SEQLENS)
        k_fp8 = k_fp8[:, :, :real_skv, :].contiguous()
        v_fp8 = v_fp8[:, :, :real_skv, :].contiguous()

    logger.info("[INFO] q_fp8 shape: %s, dtype: %s", q_fp8.shape, q_fp8.dtype)
    logger.info("[INFO] k_fp8 shape: %s, dtype: %s", k_fp8.shape, k_fp8.dtype)
    logger.info("[INFO] v_fp8 shape: %s, dtype: %s", v_fp8.shape, v_fp8.dtype)
    logger.info("[INFO] QK head dim (nope+rope): %d", qk_head_dim)
    logger.info(
        "[INFO] deq_k (per-tensor) shape: %s, val: %s",
        dequant_scale_k.shape,
        dequant_scale_k,
    )
    logger.info(
        "[INFO] deq_v (per-tensor) shape: %s, val: %s",
        dequant_scale_v.shape,
        dequant_scale_v,
    )

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32, device=device).contiguous()

    # rope bf16 (用于 CPU golden 累加和缓存, 与 nope 已合并到 576 维 fp8 不同)
    qr_bf16 = q_rope_fp16.to(torch.bfloat16).contiguous()
    kr_bf16 = k_rope_fp16.to(torch.bfloat16).contiguous()

    return (
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        qr_bf16,
        kr_bf16,
    )


# ==============================================================================
# NPU 排布 → BNSD 转换
# ==============================================================================
def _ntd_to_bnsd(q_ntd, seqused_q, n_q):
    """NTD (N_q, T, D) → BNSD (B, N_q, max_Sq, D)"""
    b = len(seqused_q)
    max_sq = max(seqused_q)
    q_bnsd = torch.zeros(
        (b, n_q, max_sq, q_ntd.shape[-1]), dtype=q_ntd.dtype, device=q_ntd.device
    )
    offset = 0
    for b_idx in range(b):
        act = seqused_q[b_idx]
        for n in range(n_q):
            q_bnsd[b_idx, n, :act, :] = q_ntd[n, offset : offset + act, :]
        offset += act
    return q_bnsd


def _bnbd_to_bnsd(kv_bnbd, block_table, cache_seqlens, block_size):
    """BNBD (block_num, N_kv, block_size, D) → BNSD (B, N_kv, max_Skv, D)"""
    b = len(cache_seqlens)
    n_kv = kv_bnbd.shape[1]
    d_dim = kv_bnbd.shape[-1]
    max_skv = max(max(cache_seqlens), 1)
    kv_bnsd = torch.zeros(
        (b, n_kv, max_skv, d_dim), dtype=kv_bnbd.dtype, device=kv_bnbd.device
    )

    for b_idx in range(b):
        seq_len = cache_seqlens[b_idx]
        block_num_per_seq = math.ceil(seq_len / block_size)
        for blk_idx in range(block_num_per_seq):
            block_id = int(block_table[b_idx, blk_idx])
            if block_id < 0:
                continue
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_len)
            valid = end_s - start_s
            if valid <= 0:
                continue
            kv_bnsd[b_idx, :, start_s:end_s, :] = kv_bnbd[block_id, :, :valid, :]
    return kv_bnsd


def _nt_to_bns1(q_scale_nt, seqused_q, n_q):
    """NT (N_q, T) → BNS1 (B, N_q, max_Sq, 1)"""
    b = len(seqused_q)
    max_sq = max(seqused_q)
    q_scale_bns1 = torch.zeros(
        (b, n_q, max_sq, 1), dtype=torch.float32, device=q_scale_nt.device
    )
    offset = 0
    for b_idx in range(b):
        act = seqused_q[b_idx]
        for n in range(n_q):
            q_scale_bns1[b_idx, n, :act, 0] = q_scale_nt[n, offset : offset + act]
        offset += act
    return q_scale_bns1


def _bnb_to_bns1(k_scale_bnb, block_table, cache_seqlens, block_size):
    """BNB (block_num, N_kv, block_size) → BNS1 (B, N_kv, max_Skv, 1)"""
    b = len(cache_seqlens)
    n_kv = k_scale_bnb.shape[1]
    max_skv = max(max(cache_seqlens), 1)
    k_scale_bns1 = torch.zeros(
        (b, n_kv, max_skv, 1), dtype=torch.float32, device=k_scale_bnb.device
    )

    for b_idx in range(b):
        seq_len = cache_seqlens[b_idx]
        block_num_per_seq = math.ceil(seq_len / block_size)
        for blk_idx in range(block_num_per_seq):
            block_id = int(block_table[b_idx, blk_idx])
            if block_id < 0:
                continue
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_len)
            valid = end_s - start_s
            if valid <= 0:
                continue
            k_scale_bns1[b_idx, :, start_s:end_s, 0] = k_scale_bnb[block_id, :, :valid]
    return k_scale_bns1


def _pa_layout_to_bnbd(kv_pa, kv_layout, n_kv=None):
    """将任意 kv_layout 的 PA cache 还原为 BnNBsD (Bn, N, Bs, D) 形式

    与 _pa_layout_transform 互逆
    BnBsND 形状为 (Bn, Bs, N, D)
    """
    if kv_layout == "BnNBsD":
        return kv_pa.contiguous()
    elif kv_layout == "BnBsND":
        return kv_pa.transpose(1, 2).contiguous()
    elif kv_layout == "NZ":
        bn, n, d1, bs, d0 = kv_pa.shape
        d = d1 * d0
        return kv_pa.permute(0, 1, 3, 2, 4).reshape(bn, n, bs, d).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def pa_cache_to_bnsd(
    k_pa,
    v_pa,
    block_table,
    cache_seqlens,
    block_size,
    kv_layout="BnNBsD",
    n_kv=None,
    k_rope_pa=None,
):
    """从 PA cache 还原 BNSD 格式的 K/V (纯数据 block), 用于 NUM_BLOCKS 非默认时做 CPU Golden 对比

    MLA K/V per-tensor: deq_k/deq_v 是独立标量, 不在 cache 中
    k_rope_pa: 可选的 rope PA cache (bf16), 同样受物理块复用影响, 需还原
    返回 (k_bnsd, v_bnsd, kr_bnsd);  k_bnsd 为 nope+rope 合并的 576 维, v_bnsd 为 nope 512 维,
    kr_bnsd 为 None 表示无 rope
    """
    k_bnbd = _pa_layout_to_bnbd(k_pa, kv_layout, n_kv=n_kv)
    v_bnbd = _pa_layout_to_bnbd(v_pa, kv_layout, n_kv=n_kv)
    k_data = k_bnbd[:, :, :block_size, :].contiguous().float()
    v_data = v_bnbd[:, :, :block_size, :].contiguous().float()

    k_bnsd = _bnbd_to_bnsd(k_data, block_table, cache_seqlens, block_size)
    v_bnsd = _bnbd_to_bnsd(v_data, block_table, cache_seqlens, block_size)

    kr_bnsd = None
    if k_rope_pa is not None:
        kr_bnbd = _pa_layout_to_bnbd(k_rope_pa, kv_layout, n_kv=n_kv)
        kr_data = kr_bnbd[:, :, :block_size, :].contiguous().float()
        kr_bnsd = _bnbd_to_bnsd(kr_data, block_table, cache_seqlens, block_size)

    return k_bnsd, v_bnsd, kr_bnsd


# ==============================================================================
# CPU Golden 函数
# MLA: bmm1 = Q_nope@K_nope^T (fp8 dequant) + Q_rope@K_rope^T (bf16)
#      softmax_scale = 1/sqrt(D + D_rope)
#      bmm2 = P @ V (V fp8 dequant, V per-tensor scale)
# ==============================================================================
def get_softmax_scale(scale_value, head_dim):
    if scale_value is not None:
        return float(scale_value)
    return 1.0 / math.sqrt(head_dim)


def torch_broadcast_kv(num_heads, num_kv_heads, tensor):
    if num_heads == num_kv_heads:
        return tensor.contiguous()
    factor = num_heads // num_kv_heads
    return tensor.repeat_interleave(factor, dim=1).contiguous()


# ==============================================================================
# Kernel 分段信息解码
# NPU kernel 按 (bn, m-block) 为单位将 s2 范围切分为多个分段 (分核/分 section),
# 每个分段独立维护 running max 并以 P*fp8max 量化 P, 段间按 FD 语义合并。
# golden 需按相同分段模拟才能对齐 fp8 舍入网格。
# ==============================================================================
_QMLA_SECTION_CACHE = {}


def get_qmla_section_info(seqused_q, cache_seqlens):
    """调用 metadata 算子并解码 FA 分核分段信息, 返回 (m_base, s2_base, segments)

    segments[bn] = [ [(s2块起, s2块止), ...] 每个 m-block 的分段列表 ]
    解码失败时返回 None, golden 退化为全局量化语义。
    """
    key = (
        tuple(int(s) for s in seqused_q),
        tuple(int(s) for s in cache_seqlens),
        N_q,
        N_kv,
        MASK_MODE,
        INPUT_LAYOUT,
    )
    if key in _QMLA_SECTION_CACHE:
        return _QMLA_SECTION_CACHE[key]
    info = _build_qmla_section_info(seqused_q, cache_seqlens)
    _QMLA_SECTION_CACHE[key] = info
    return info


def _build_qmla_section_info(seqused_q, cache_seqlens):
    if quant_flash_mla_with_kvcache_metadata is None:
        logger.warning("[Section] metadata 算子不可用, golden 使用全局量化语义")
        return None
    try:
        cache_seqlens_t = torch.tensor(
            cache_seqlens, dtype=torch.int32, device="npu"
        ).contiguous()
        # 非 TND (BNSD/BSND) 时 op 不支持 cu_seqlens_q, 传 None
        if INPUT_LAYOUT == "TND":
            cu_q = [0]
            acc = 0
            for s in seqused_q:
                acc += int(s)
                cu_q.append(acc)
            cu_seqlens_q = torch.tensor(
                cu_q, dtype=torch.int32, device="npu"
            ).contiguous()
        else:
            cu_seqlens_q = None
        seqused_q_t = torch.tensor(
            seqused_q, dtype=torch.int32, device="npu"
        ).contiguous()
        max_sq = MAX_SEQLEN_Q if MAX_SEQLEN_Q > 0 else max(seqused_q)
        max_skv = MAX_SEQLEN_KV if MAX_SEQLEN_KV > 0 else max(cache_seqlens)
        meta = quant_flash_mla_with_kvcache_metadata(
            cache_seqlens=cache_seqlens_t,
            num_heads_q=N_q,
            num_heads_kv=N_kv,
            quant_mode=QMLA_QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q_t,
            max_seqlen_q=max_sq,
            max_seqlen_kv=max_skv,
            head_dim_qk=QMLA_QK_HEAD_DIM,
            head_dim_v=QMLA_V_HEAD_DIM,
            mask_mode=MASK_MODE,
            layout_q=INPUT_LAYOUT,
        )
        torch.npu.synchronize()
        meta_flat = meta.contiguous().cpu().view(-1).tolist()
        return _decode_qmla_sections(meta_flat, seqused_q, cache_seqlens)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[Section] metadata 调用/解码失败(%s), golden 使用全局量化语义", exc
        )
        return None


def _decode_qmla_sections(meta_flat, seqused_q, cache_seqlens):
    """解码 metadata: header[16] + FA区[section][aicCore(36)][16] + FD区

    每个 FA chunk 是 (bn, m, s2) 线性迭代空间上的一段 [start, end),
    m 轴为 GS1 合轴: bn 内 row = s1 * group + g。
    """
    AIC_CORE_NUM = 36
    META_SIZE = 16
    HEAD_SIZE = 16
    section_num = int(meta_flat[0])
    m_base = int(meta_flat[2])
    s2_base = int(meta_flat[3])
    if section_num <= 0 or m_base <= 0 or s2_base <= 0:
        raise ValueError(
            f"invalid metadata header: section={section_num}, mBase={m_base}, s2Base={s2_base}"
        )

    batch = len(cache_seqlens)
    bn_total = batch * N_kv
    group = N_q // N_kv
    m_blocks = [
        (int(seqused_q[b]) * group + m_base - 1) // m_base for b in range(batch)
    ]
    s2_blocks = [(int(cache_seqlens[b]) + s2_base - 1) // s2_base for b in range(batch)]
    segments = [[[] for _ in range(m_blocks[b // N_kv])] for b in range(bn_total)]

    for sec in range(section_num):
        for core in range(AIC_CORE_NUM):
            base_idx = HEAD_SIZE + (sec * AIC_CORE_NUM + core) * META_SIZE
            bn_s, m_s, s2_s, bn_e, m_e, s2_e = (
                int(v) for v in meta_flat[base_idx : base_idx + 6]
            )
            if bn_s == bn_e and m_s == m_e and s2_s == s2_e:
                continue
            bn, mi, s2 = bn_s, m_s, s2_s
            steps = 0
            while (bn, mi, s2) != (bn_e, m_e, s2_e):
                steps += 1
                if steps > 100000000 or bn >= bn_total:
                    raise ValueError("metadata chunk walk 越界")
                b_idx = bn // N_kv
                row_end = s2_blocks[b_idx]
                m_end = m_blocks[b_idx]
                if mi >= m_end or row_end == 0:
                    mi += 1
                    s2 = 0
                    if mi >= m_end:
                        mi = 0
                        bn += 1
                    continue
                if bn == bn_e and mi == m_e:
                    seg_end = min(s2_e, row_end)
                else:
                    seg_end = row_end
                if seg_end > s2:
                    segments[bn][mi].append((s2, seg_end))
                if seg_end >= row_end:
                    mi += 1
                    s2 = 0
                    if mi >= m_end:
                        mi = 0
                        bn += 1
                else:
                    s2 = seg_end

    # 覆盖率校验: 每个 m-block 的分段必须完整覆盖其 s2 范围
    for bn in range(bn_total):
        b_idx = bn // N_kv
        for mi in range(m_blocks[b_idx]):
            cover = sum(bb - aa for aa, bb in segments[bn][mi])
            if cover != s2_blocks[b_idx]:
                raise ValueError(
                    f"bn={bn} m={mi} 分段覆盖 {cover} != {s2_blocks[b_idx]}"
                )
    return m_base, s2_base, segments


def _fp8_fullquant_mla_golden_sectioned(
    q_tensor,
    k_tensor,
    v_tensor,
    deq_q,
    deq_k,
    deq_v,
    seqused_q,
    cache_seqlens,
    section_info,
):
    """分段 golden: 按 kernel 分段调度模拟 online softmax 与 P 量化

    每个 (bn, m-block) 的各 s2 分段独立维护 running max, P 以 P*fp8max 量化,
    分段结果按 FD 语义 (max 重标定) 合并, 与 NPU isMlaFullQuant 路径一致。
    """
    softmax_scale = get_softmax_scale(SCALE_VALUE, D)
    m_base, s2_base, segments = section_info
    batch, heads, q_seq, _ = q_tensor.shape
    v_dim = v_tensor.shape[-1]
    group = heads // N_kv
    minValue = -3.402823466e38
    fp8_e4m3_max = 448.0
    deq_k_scalar = deq_k.reshape(-1)[0] if deq_k.numel() == 1 else deq_k
    deq_v_scalar = deq_v.reshape(-1)[0] if deq_v.numel() == 1 else deq_v

    result = torch.zeros(
        (batch, heads, q_seq, v_dim), dtype=torch.float32, device=q_tensor.device
    )
    lse = torch.full(
        (batch, heads, q_seq, 1),
        float("inf"),
        dtype=torch.float32,
        device=q_tensor.device,
    )

    for b in range(batch):
        q_len = int(seqused_q[b])
        kv_len = int(cache_seqlens[b])
        for n2 in range(N_kv):
            bn = b * N_kv + n2
            k_bn = k_tensor[b, n2]
            v_bn = v_tensor[b, n2]
            for mi, segs in enumerate(segments[bn]):
                r0 = mi * m_base
                r1 = min((mi + 1) * m_base, q_len * group)
                if r1 <= r0:
                    continue
                rows = torch.arange(r0, r1, device=q_tensor.device)
                # m 轴行序与 kernel GM 格式一致: TND/BSND(BSNGD) 为 S1G 序 row=s1*group+g;
                # BNSD(BNGSD) 为 GS1 序 row=g*s1+s1
                if INPUT_LAYOUT == "BNSD":
                    g_idx = (rows // q_len).long()
                    s1_idx = (rows % q_len).long()
                else:
                    s1_idx = (rows // group).long()
                    g_idx = (rows % group).long()
                head_idx = n2 * group + g_idx
                q_sel = q_tensor[b, head_idx, s1_idx, :]
                dq_sel = deq_q[b, head_idx, s1_idx, 0]
                dq_scaled = (dq_sel * softmax_scale)[:, None]

                o_g = torch.zeros(
                    (r1 - r0, v_dim), dtype=torch.float32, device=q_tensor.device
                )
                s_g = torch.zeros(
                    (r1 - r0, 1), dtype=torch.float32, device=q_tensor.device
                )
                m_g = torch.full(
                    (r1 - r0, 1), minValue, dtype=torch.float32, device=q_tensor.device
                )

                for blk_a, blk_b in segs:
                    o = torch.zeros_like(o_g)
                    s = torch.zeros_like(s_g)
                    mx = torch.full_like(m_g, minValue)
                    for j in range(blk_a, blk_b):
                        lo = j * s2_base
                        kj = k_bn[lo : lo + s2_base]
                        vj = v_bn[lo : lo + s2_base]
                        n_col = kj.shape[0]
                        sij = torch.matmul(q_sel, kj.transpose(0, 1))
                        sij = sij * dq_scaled * deq_k_scalar
                        cols = torch.arange(lo, lo + n_col, device=q_tensor.device)
                        col_mask = cols >= kv_len
                        if MASK_MODE == 3:
                            causal = cols[None, :] > (s1_idx[:, None] + kv_len - q_len)
                            col_mask = col_mask[None, :] | causal
                        sij = sij.masked_fill(col_mask, float("-inf"))

                        m_block, _ = torch.max(sij, dim=-1, keepdims=True)
                        mx_old = mx.clone()
                        mx_new = torch.maximum(m_block, mx)
                        upd = torch.exp(mx_old - mx_new)
                        upd = torch.where(
                            mx_old <= minValue, torch.zeros_like(upd), upd
                        )
                        pij = torch.exp(sij - mx_new)
                        pij = torch.where(
                            sij == float("-inf"), torch.zeros_like(pij), pij
                        )
                        s_blk = torch.sum(pij, dim=-1, keepdims=True)
                        # MLA rescale: P 直接乘 fp8 max 量化 (无 rowmax_p 重标定)
                        pij_scaled = (
                            (pij * fp8_e4m3_max).to(FP8_DTYPE).to(torch.float32)
                        )
                        bmm2_res = torch.matmul(pij_scaled, vj)
                        o = o * upd + bmm2_res
                        s = s * upd + s_blk
                        mx = mx_new
                    # 分段合并 (FD 语义: max 重标定累加)
                    mg_new = torch.maximum(m_g, mx)
                    sc_g = torch.exp(m_g - mg_new)
                    sc_g = torch.where(m_g <= minValue, torch.zeros_like(sc_g), sc_g)
                    sc_s = torch.exp(mx - mg_new)
                    sc_s = torch.where(mx <= minValue, torch.zeros_like(sc_s), sc_s)
                    o_g = o_g * sc_g + o * sc_s
                    s_g = s_g * sc_g + s * sc_s
                    m_g = mg_new

                all_masked = (m_g <= minValue).squeeze(-1)
                denom = s_g + EPSILON
                res_rows = o_g / denom / fp8_e4m3_max * deq_v_scalar
                res_rows = torch.where(
                    all_masked[:, None], torch.zeros_like(res_rows), res_rows
                )
                result[b, head_idx, s1_idx, :] = res_rows
                lse_rows = torch.where(
                    all_masked,
                    torch.full_like(
                        all_masked,
                        float("inf"),
                        dtype=torch.float32,
                        device=q_tensor.device,
                    ),
                    (m_g + torch.log(denom)).squeeze(-1),
                )
                lse[b, head_idx, s1_idx, 0] = lse_rows
    return result.contiguous(), lse.contiguous()


def fp8_fullquant_mla_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    deq_q,
    deq_k,
    deq_v,
    p_scale,
    seqused_q,
    cache_seqlens,
    qr_bf16=None,
    kr_bf16=None,
    *,
    golden_device="npu",
):
    """Device-selectable golden reference - MLA flash attention with FP8 quantization

    Q/K: nope(512)+rope(64) 已合并为 576 维, 统一 fp8 per-token-head/per-tensor 量化,
         反量化后直接做一次 576 维 matmul (无需单独的 rope 累加)
    V:   使用 K 的 nope 前 512 维 (per-tensor 量化), bmm2 = P @ V
    softmax_scale = 1/sqrt(D) (仅 nope 部分, 不含 rope)
    """
    softmax_scale = get_softmax_scale(SCALE_VALUE, D)

    device = torch.device(golden_device)
    logger.info("[Golden] compute device=%s", device)
    q_tensor = q_fp8.to(device=device, dtype=torch.float32).contiguous()
    k_tensor = k_fp8.to(device=device, dtype=torch.float32).contiguous()
    v_tensor = v_fp8.to(device=device, dtype=torch.float32).contiguous()
    deq_q = deq_q.to(device=device, dtype=torch.float32).contiguous()
    deq_k = deq_k.to(device=device, dtype=torch.float32).contiguous()
    deq_v = deq_v.to(device=device, dtype=torch.float32).contiguous()

    batch, heads, q_seq, d_dim = q_tensor.shape
    v_dim = v_tensor.shape[-1]

    # 空 KV 场景
    if k_tensor.shape[2] == 0:
        result = torch.zeros(
            (batch, heads, q_seq, v_dim), dtype=torch.float32, device=device
        ).contiguous()
        lse = torch.full(
            (batch, heads, q_seq, 1), float("inf"), dtype=torch.float32, device=device
        ).contiguous()
        return result, lse

    # 对齐 kernel 分段量化语义: 解码 metadata 分核分段, 按 (bn, m-block) 分段模拟
    section_info = get_qmla_section_info(seqused_q, cache_seqlens)
    if section_info is not None:
        return _fp8_fullquant_mla_golden_sectioned(
            q_tensor,
            k_tensor,
            v_tensor,
            deq_q,
            deq_k,
            deq_v,
            seqused_q,
            cache_seqlens,
            section_info,
        )

    out = torch.zeros(
        (batch, heads, q_seq, v_dim), dtype=torch.float32, device=q_tensor.device
    ).contiguous()
    o_sum = torch.zeros(
        q_tensor.shape[:-1], dtype=torch.float32, device=q_tensor.device
    )[..., None].contiguous()
    minValue = torch.tensor(
        -3.402823466e38, dtype=torch.float32, device=q_tensor.device
    )
    o_max = torch.full(
        q_tensor.shape[:-1],
        minValue.item(),
        dtype=torch.float32,
        device=q_tensor.device,
    )[..., None].contiguous()

    q_lens_t = torch.tensor(seqused_q, dtype=torch.int32, device=device).contiguous()
    k_lens_t = torch.tensor(
        cache_seqlens, dtype=torch.int32, device=device
    ).contiguous()
    q_lens_acl = q_lens_t.view(batch, 1, 1, 1).contiguous()
    k_lens_acl = k_lens_t.view(batch, 1, 1, 1).contiguous()

    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]
    q_range = torch.arange(Sq, device=device).view(1, 1, -1, 1).contiguous()
    k_range = torch.arange(Skv, device=device).view(1, 1, 1, -1).contiguous()
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if MASK_MODE == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        mask_global = causal_mask | q_padding_mask | k_padding_mask
    else:
        mask_global = q_padding_mask | k_padding_mask
    mask_global = mask_global.contiguous()

    mask_q_blocks = list(torch.split(mask_global, Q_BLOCK_SIZE, dim=2))
    mask_blocks = []
    for mask_q_block in mask_q_blocks:
        mask_blocks.append(list(torch.split(mask_q_block, KV_BLOCK_SIZE, dim=3)))

    q_blocks = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    k_blocks = list(torch.split(k_tensor, KV_BLOCK_SIZE, dim=2))
    v_blocks = list(torch.split(v_tensor, KV_BLOCK_SIZE, dim=2))
    o_blocks = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_blocks = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_blocks = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))
    deq_q_blocks = list(torch.split(deq_q, Q_BLOCK_SIZE, dim=2))

    # K/V per-tensor: deq_k/deq_v 是标量
    deq_k_scalar = deq_k.reshape(-1)[0] if deq_k.numel() == 1 else deq_k
    deq_v_scalar = deq_v.reshape(-1)[0] if deq_v.numel() == 1 else deq_v
    fp8_e4m3_max = 448.0

    for j, (kj, vj) in enumerate(zip(k_blocks, v_blocks)):
        # Broadcast only this KV block, avoiding a full sequence copy per Q head.
        kj = torch_broadcast_kv(N_q, N_kv, kj)
        vj = torch_broadcast_kv(N_q, N_kv, vj)
        kj_T = kj.transpose(-1, -2).contiguous()
        vj = vj.contiguous()

        for i, qi in enumerate(q_blocks):
            oi = o_blocks[i]
            si = s_blocks[i]
            mi = m_blocks[i]
            deq_qi = deq_q_blocks[i]

            # bmm1: 576 维 (nope+rope) 合并的 Q@K^T, 整体乘 deq_q * deq_k * scalar
            sij = torch.matmul(qi, kj_T)
            deq_qi_scaled = deq_qi * softmax_scale
            sij = sij * deq_qi_scaled * deq_k_scalar

            causal_mask = mask_blocks[i][j].contiguous()
            sij = sij.masked_fill(causal_mask, float("-inf"))

            # online softmax: rowmax, update_mul, rowSum
            m_block, _ = torch.max(sij, dim=-1, keepdims=True)
            mi_old = mi.clone()
            mi_new = torch.maximum(m_block, mi)
            update_mul = torch.exp(mi_old - mi_new)
            update_mul = torch.where(
                mi_old <= minValue.item(), torch.zeros_like(update_mul), update_mul
            )

            pij = torch.exp(sij - mi_new)
            pij = torch.where(sij == float("-inf"), torch.zeros_like(pij), pij)
            s_block = torch.sum(pij, dim=-1, keepdims=True)

            # MLA rescale: P 直接乘 fp8 max 量化 (无 rowmax_p 重标定, 与 NPU isMlaFullQuant=true 一致)
            pij_scaled = (pij * fp8_e4m3_max).to(FP8_DTYPE).to(torch.float32)

            # bmm2: P_fp8 @ V (deq_v 在最终输出时统一乘)
            bmm2_res = torch.matmul(pij_scaled, vj)

            # O_flash 累加: O = O * update_mul + bmm2_res
            o_blocks[i] = oi * update_mul + bmm2_res
            s_blocks[i] = si * update_mul + s_block
            m_blocks[i] = mi_new

    # 最终输出: O = O_flash / rowSum / 448 * deq_v
    result = torch.cat(o_blocks, dim=2).contiguous()
    out_sum = torch.cat(s_blocks, dim=2).contiguous()
    out_max = torch.cat(m_blocks, dim=2).contiguous()

    result = result / (out_sum + EPSILON) / fp8_e4m3_max * deq_v_scalar

    all_masked = out_max <= minValue.item()
    lse = torch.where(
        all_masked,
        torch.full_like(out_max, float("inf")),
        out_max + torch.log(out_sum + EPSILON),
    ).contiguous()
    result = torch.where(all_masked, torch.zeros_like(result), result)
    return result, lse


def cpu_fp8_fullquant_mla_golden(*args, **kwargs):
    """Backward-compatible CPU reference entry point."""
    kwargs["golden_device"] = "cpu"
    return fp8_fullquant_mla_golden(*args, **kwargs)


# ==============================================================================
# Layout 转换
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """BNSD → 各种 layout"""
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    tensor = tensor.contiguous()
    b, n, _, d = tensor.shape
    max_org_s = max(seq_lens)

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return (
            tensor[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(b, max_org_s, n * d)
            .contiguous()
        )
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, n, d), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b_idx in range(b):
            act_s = seq_lens[b_idx]
            for n_idx in range(n):
                result[t : t + act_s, n_idx, :] = tensor[b_idx, n_idx, :act_s, :]
            t += act_s
        return result.contiguous()
    elif layout == "NTD_TND":
        T = sum(seq_lens)
        result = torch.zeros((n, T, d), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b_idx in range(b):
            act_s = seq_lens[b_idx]
            for n_idx in range(n):
                result[n_idx, t : t + act_s, :] = tensor[b_idx, n_idx, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_scale_to_layout(tensor, seq_lens, scale_type):
    """Scale to layout"""
    tensor = tensor.contiguous()
    if scale_type == "deq_q":
        b, n, _, _ = tensor.shape
        T = sum(seq_lens)
        # Q scale layout 由 INPUT_LAYOUT 自动推导 (不暴露 Q_SCALE_LAYOUT 字段)
        q_scale_layout = _Q_SCALE_LAYOUT_MAP.get(INPUT_LAYOUT, "BNS")
        if q_scale_layout == "NT":
            result = torch.zeros((n, T), dtype=torch.float32, device=tensor.device)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[n_idx, t : t + act_s] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif q_scale_layout == "TN":
            result = torch.zeros((T, n), dtype=torch.float32, device=tensor.device)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[t : t + act_s, n_idx] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif q_scale_layout == "BNS":
            # BNSD→BNS: S 取自 tensor 第 2 维, 不依赖 seq_lens
            s_dim = tensor.shape[2]
            result = torch.zeros(
                (b, n, s_dim), dtype=torch.float32, device=tensor.device
            )
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[b_idx, n_idx, :act_s] = tensor[b_idx, n_idx, :act_s, 0]
            return result.contiguous()
        elif q_scale_layout == "BSN":
            s_dim = tensor.shape[2]
            result = torch.zeros(
                (b, s_dim, n), dtype=torch.float32, device=tensor.device
            )
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[b_idx, :act_s, n_idx] = tensor[b_idx, n_idx, :act_s, 0]
            return result.contiguous()
        else:
            return tensor.float().contiguous()
    elif scale_type == "deq_v":
        # V per-tensor: 标量, 直接返回 1D
        return tensor.reshape(tensor.numel()).float().contiguous()
    return tensor.squeeze(-1).contiguous()


def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


# ==============================================================================
# NPU 调用
# ==============================================================================
def get_npu_fa_kwargs():
    return dict(
        query_quant_mode=3,  # per-token-head
        key_quant_mode=0,  # per-tensor
        value_quant_mode=0,  # per-tensor
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        return_softmax_lse=ENABLE_LSE,
    )


class Network(nn.Module):
    """aclgraph/torch.compile 编译目标: forward 只包含两个 torch.library op 调用。

    所有预处理（tensor 创建、reshape、layout 转换）在 fia_mla_torch_npu 中完成，
    传入 forward 的均为已处理好的 NPU tensor 或标量。
    """

    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q_576,
        k_cache_576,
        deq_q_flat,
        deq_k_scalar,
        block_table,
        cache_seqlens_t,
        metadata,
        cu_seqlens_q,
        seqused_q_t,
        mask,
        softmax_scale,
        layout,
        layout_kv,
        max_sq,
        max_skv,
    ):
        # metadata 在图外算好作为输入 tensor 传入:
        # metadata 的 register_fake 硬编码 meta device, 若在图内调用会与
        # npu FakeTensor 混用触发 Unhandled FakeTensor Device Propagation
        atten_out, lse_out = (
            torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache(
                q_576,
                k_cache_576,
                deq_q_flat,
                deq_k_scalar,
                block_table,
                cache_seqlens_t,
                QMLA_QUANT_MODE,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=seqused_q_t,
                attn_mask=mask,
                metadata=metadata,
                softmax_scale=softmax_scale,
                mask_mode=MASK_MODE,
                max_seqlen_q=max_sq,
                max_seqlen_kv=max_skv,
                head_dim_v=QMLA_V_HEAD_DIM,
                layout_q=layout,
                layout_kv=layout_kv,
                layout_out=layout,
                return_softmax_lse=ENABLE_LSE,
            )
        )
        return atten_out, lse_out


class QmlaGraphNetwork(nn.Module):
    """GRAPH_PATH=8 (aclgraph+静态kernel) 编译目标。

    metadata 在图外算好作为输入 tensor 传入, forward 仅主算子原子调用,
    对齐 flash_attn npu.py 的 FlashAttnGraphNetwork 结构。
    """

    def __init__(self):
        super(QmlaGraphNetwork, self).__init__()

    def forward(
        self,
        q_576,
        k_cache_576,
        deq_q_flat,
        deq_k_scalar,
        block_table,
        cache_seqlens_t,
        metadata,
        cu_seqlens_q,
        seqused_q_t,
        mask,
        softmax_scale,
        layout,
        layout_kv,
        max_sq,
        max_skv,
    ):
        atten_out, lse_out = (
            torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache(
                q_576,
                k_cache_576,
                deq_q_flat,
                deq_k_scalar,
                block_table,
                cache_seqlens_t,
                QMLA_QUANT_MODE,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=seqused_q_t,
                attn_mask=mask,
                metadata=metadata,
                softmax_scale=softmax_scale,
                mask_mode=MASK_MODE,
                max_seqlen_q=max_sq,
                max_seqlen_kv=max_skv,
                head_dim_v=QMLA_V_HEAD_DIM,
                layout_q=layout,
                layout_kv=layout_kv,
                layout_out=layout,
                return_softmax_lse=ENABLE_LSE,
            )
        )
        return atten_out, lse_out


def call_npu_fa_op(
    q,
    k,
    mask,
    seqused_q,
    cache_seqlens,
    dequant_scale_q,
    dequant_scale_k,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    block_size,
    out_dtype,
):
    """新接口两段式调用：metadata -> 主算子 quant_flash_mla_with_kvcache

    :param q:        Q (T, N_q, 576) fp8, nope(512)+rope(64) 已合并 (fa_run_npu 传入)
    :param k:        K_cache PA (Bn, N, Bs, 576) fp8, 与 Q 相同合并方式
    V 由算子内部从 k_cache 的 nope 前 512 维复用, 无需传入
    """
    assert quant_flash_mla_with_kvcache is not None, (
        "cann_ops_transformer not installed"
    )
    torch.npu.synchronize()

    # ---- 1) q / k_cache 已是 nope+rope 合并后的 576 维 fp8, 直接使用 ----
    q_576 = q.to(FP8_DTYPE).contiguous()
    # k_cache 保持原 stride (非连续 view 不做 contiguous 拷贝), 由算子按 tiling stride 寻址
    k_cache_576 = k if k.dtype == FP8_DTYPE else k.to(FP8_DTYPE)

    # ---- 2) per-token-head Q descale: TND -> (T, N_q) 2D; BNSD/BSND -> 3D (B,N,S)/(B,S,N) ----
    deq_q_flat = dequant_scale_q.float().contiguous()
    if layout == "TND":
        if deq_q_flat.dim() == 3:  # (T, N, 1) -> (T, N)
            deq_q_flat = deq_q_flat.squeeze(-1)
        if deq_q_flat.dim() == 4:  # (B, N, S, 1) -> TND (T, N)
            b, n, s, _ = deq_q_flat.shape
            deq_q_flat = deq_q_flat.permute(0, 2, 1, 3).reshape(b * s, n).contiguous()
    else:
        # BNSD/BSND: op 要求 q_descale 3D, (B, N, S, 1)/(B, S, N, 1) -> squeeze 末维
        if deq_q_flat.dim() == 4:
            deq_q_flat = deq_q_flat.squeeze(-1).contiguous()
    deq_q_flat = deq_q_flat.float().npu().contiguous()

    deq_k_scalar = (
        dequant_scale_k.float().reshape(-1)[0].contiguous().view(1).view(-1).npu()
    )

    # cache_seqlens: PA场景下为每batch KV实际长度（BY_BATCH模式, 非累积）
    max_sq = MAX_SEQLEN_Q if MAX_SEQLEN_Q > 0 else max(seqused_q)
    max_skv = MAX_SEQLEN_KV if MAX_SEQLEN_KV > 0 else max(cache_seqlens)
    cache_seqlens_t = torch.tensor(
        cache_seqlens, dtype=torch.int32, device="npu"
    ).contiguous()
    # cu_seqlens_q (TND): 累计序列，含前导 0 -> [0, l1, l1+l2, ...]
    # 非 TND (BNSD/BSND) 时 op 不支持 cu_seqlens_q, 传 None
    if layout == "TND":
        cu_q = [0]
        _acc = 0
        for _s in seqused_q:
            _acc += int(_s)
            cu_q.append(_acc)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device="npu").contiguous()
    else:
        cu_seqlens_q = None
    # seqused_q: 每 batch 实际 Q 长度（非累计）
    seqused_q_t = torch.tensor(seqused_q, dtype=torch.int32, device="npu").contiguous()
    # 确保device数据同步完成后再传入metadata算子
    torch.npu.synchronize()
    # ---- 3) 两段式：先 metadata，再主算子 ----
    metadata = quant_flash_mla_with_kvcache_metadata(
        cache_seqlens=cache_seqlens_t,
        num_heads_q=q_n,
        num_heads_kv=kv_n,
        quant_mode=QMLA_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q_t,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
        head_dim_qk=QMLA_QK_HEAD_DIM,
        head_dim_v=QMLA_V_HEAD_DIM,
        mask_mode=MASK_MODE,
        layout_q=layout,
    )

    # Reuse the exact prepared inputs and metadata for profiling repetitions.
    block_table_npu = block_table.int().npu().contiguous()
    logger.info(
        "[NPU] warmup=%d runs=%d flush_l2=%s (256 MiB read)",
        OP_WARMUP,
        OP_RUNS,
        OP_FLUSH_L2,
    )

    def _run_once(iteration):
        if OP_FLUSH_L2:
            # Match testkit core/operator.py gen_cold_body (FLUSH_MODE=read).
            flush_buf = torch.empty(
                256 * 1024 * 1024 // 4, dtype=torch.float32, device=q_576.device
            )
            flush_sum = flush_buf.sum()
            torch.npu.synchronize()
            del flush_buf, flush_sum
            torch.npu.empty_cache()
        atten_out, lse_out = quant_flash_mla_with_kvcache(
            q=q_576,
            k_cache=k_cache_576,
            q_descale=deq_q_flat,
            k_descale=deq_k_scalar,
            block_table=block_table_npu,
            cache_seqlens=cache_seqlens_t,
            quant_mode=QMLA_QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q_t,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            mask_mode=MASK_MODE,
            max_seqlen_q=max_sq,
            max_seqlen_kv=max_skv,
            head_dim_v=QMLA_V_HEAD_DIM,
            layout_q=layout,
            layout_kv=_KV_LAYOUT_MAP.get(KV_CACHE_LAYOUT, "PA_BNBD"),
            layout_out=layout,
            return_softmax_lse=ENABLE_LSE,
        )
        if OP_FLUSH_L2 or iteration + 1 == OP_WARMUP:
            torch.npu.synchronize()
        if OP_FLUSH_L2:
            torch.npu.empty_cache()
        return atten_out, lse_out

    def _run_loop():
        atten_out = lse_out = None
        for iteration in range(OP_WARMUP, OP_WARMUP + OP_RUNS):
            atten_out, lse_out = _run_once(iteration)
        return atten_out, lse_out

    if QMLA_PROF:
        # 仅 profiling 测量循环: warmup 留在外面, golden 不进 profiling
        from torch_npu.profiler import ProfilerActivity, profile as npu_profile

        for iteration in range(OP_WARMUP):
            _run_once(iteration)
        with npu_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]
        ) as prof:
            atten_out, lse_out = _run_loop()
        prof.export_chrome_trace("qmla_fa_run_trace.json")
        logger.info("[PROF] trace saved to qmla_fa_run_trace.json")
    else:
        for iteration in range(OP_WARMUP):
            _run_once(iteration)
        atten_out, lse_out = _run_loop()
    torch.npu.synchronize()
    return atten_out, lse_out


def fia_mla_torch_npu(
    q,
    k,
    mask,
    seqused_q,
    cache_seqlens,
    dequant_scale_q,
    dequant_scale_k,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    block_size,
    out_dtype,
):
    if GRAPH_PATH == 0:
        logger.info("[NPU] GRAPH_PATH == 0, 单算子模式...")
        return call_npu_fa_op(
            q,
            k,
            mask,
            seqused_q,
            cache_seqlens,
            dequant_scale_q,
            dequant_scale_k,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout,
            block_size,
            out_dtype,
        )

    # ---- 预处理（compile 区域外）----
    torch.npu.synchronize()
    q_576 = q.to(FP8_DTYPE).contiguous()
    k_cache_576 = k.to(FP8_DTYPE).contiguous()

    # Q descale: TND -> (T, N_q) 2D; BNSD/BSND -> 3D (B,N,S)/(B,S,N)
    deq_q_flat = dequant_scale_q.float().contiguous()
    if layout == "TND":
        if deq_q_flat.dim() == 3:
            deq_q_flat = deq_q_flat.squeeze(-1)
        if deq_q_flat.dim() == 4:
            b, n, s, _ = deq_q_flat.shape
            deq_q_flat = deq_q_flat.permute(0, 2, 1, 3).reshape(b * s, n).contiguous()
    else:
        if deq_q_flat.dim() == 4:
            deq_q_flat = deq_q_flat.squeeze(-1).contiguous()
    deq_q_flat = deq_q_flat.float().npu().contiguous()

    deq_k_scalar = (
        dequant_scale_k.float().reshape(-1)[0].contiguous().view(1).view(-1).npu()
    )

    max_sq = MAX_SEQLEN_Q if MAX_SEQLEN_Q > 0 else max(seqused_q)
    max_skv = MAX_SEQLEN_KV if MAX_SEQLEN_KV > 0 else max(cache_seqlens)
    cache_seqlens_t = torch.tensor(
        cache_seqlens, dtype=torch.int32, device="npu"
    ).contiguous()
    # cu_seqlens_q (TND): 累计序列，含前导 0 -> [0, l1, l1+l2, ...]
    # 非 TND (BNSD/BSND) 时 op 不支持 cu_seqlens_q, 传 None
    if layout == "TND":
        cu_q = [0]
        _acc = 0
        for _s in seqused_q:
            _acc += int(_s)
            cu_q.append(_acc)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device="npu").contiguous()
    else:
        cu_seqlens_q = None
    seqused_q_t = torch.tensor(seqused_q, dtype=torch.int32, device="npu").contiguous()
    block_table_t = (
        block_table.int().npu().contiguous()
        if not block_table.is_npu
        else block_table.int().contiguous()
    )
    layout_kv = _KV_LAYOUT_MAP.get(KV_CACHE_LAYOUT, "PA_BNBD")
    # metadata 图外算好作为图输入 (metadata 的 register_fake 为 meta device,
    # 图内调用会触发 Unhandled FakeTensor Device Propagation)
    metadata = torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache_metadata(
        cache_seqlens_t,
        N_q,
        N_kv,
        QMLA_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q_t,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
        head_dim_qk=QMLA_QK_HEAD_DIM,
        head_dim_v=QMLA_V_HEAD_DIM,
        mask_mode=MASK_MODE,
        layout_q=layout,
    )
    torch.npu.synchronize()

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    config = CompilerConfig()
    with torch.no_grad():
        torch.npu.synchronize()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        fa_args = (
            q_576,
            k_cache_576,
            deq_q_flat,
            deq_k_scalar,
            block_table_t,
            cache_seqlens_t,
            metadata,
            cu_seqlens_q,
            seqused_q_t,
            mask,
            softmax_scale,
            layout,
            layout_kv,
            max_sq,
            max_skv,
        )

        if GRAPH_PATH == 7:
            logger.info("[NPU] GRAPH_PATH == 7, aclgraph...")
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            config.mode = "reduce-overhead"
            torch._dynamo.reset()
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            for t in (
                q_576,
                k_cache_576,
                deq_q_flat,
                deq_k_scalar,
                block_table_t,
                cache_seqlens_t,
                cu_seqlens_q,
                seqused_q_t,
                mask,
            ):
                if t is not None:
                    torch._dynamo.mark_static(t)
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH == 8:
            logger.info("[NPU] GRAPH_PATH == 8, aclgraph+静态kernel...")
            # metadata 已在图外算好, 图内仅主算子原子调用 (对齐 flash_attn)
            torch._dynamo.reset()
            graph_net = QmlaGraphNetwork().to("npu:%s" % int(DEVICE_ID))
            config.mode = "reduce-overhead"
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = (
                "./"
            )
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            config.experimental_config.topology_sorting_strategy = "StableRDFS"
            npu_backend = tng.get_npu_backend(compiler_config=config)
            graph_net = torch.compile(
                graph_net, fullgraph=False, backend=npu_backend, dynamic=False
            )
            atten_out, lse_out = graph_net(
                q_576,
                k_cache_576,
                deq_q_flat,
                deq_k_scalar,
                block_table_t,
                cache_seqlens_t,
                metadata,
                cu_seqlens_q,
                seqused_q_t,
                mask,
                softmax_scale,
                layout,
                layout_kv,
                max_sq,
                max_skv,
            )
        else:
            raise ValueError(
                f"Unsupported GRAPH_PATH: {GRAPH_PATH}, only support 0/7/8"
            )

        atten_out = atten_out.detach()
        lse_out = lse_out.detach()
        torch.npu.synchronize()
        return atten_out, lse_out


def fa_run_npu(
    q,
    k,
    mask,
    seqused_q,
    cache_seqlens,
    dequant_scale_q,
    dequant_scale_k,
    p_scale,
    block_table,
    block_size,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    out_dtype,
):
    """将数据转移到NPU上并调用NPU算子"""
    torch_npu.npu.set_device(int(DEVICE_ID))

    # 非 TND 场景 (BNSD/BSND) 不传 cu_seqlens_q, seqused_q 由 tensor shape 推断 S
    if layout not in ("TND", "NTD_TND"):
        if layout == "BSND":
            seqused_q = [int(q.shape[1])] * int(q.shape[0])
        else:  # BNSD
            seqused_q = [int(q.shape[2])] * int(q.shape[0])
        cache_seqlens = None
    if ENABLE_PA:
        cache_seqlens = CACHE_SEQLENS

    q = q.npu()
    k = k.npu()

    # MLA K per-tensor: deq_k 独立传入 (标量, 不与 key 共享内存)
    dequant_scale_k = dequant_scale_k.float().npu()

    dequant_scale_q = dequant_scale_q.float().npu()
    p_scale = p_scale.float().npu()

    block_table = block_table.int().npu() if ENABLE_PA else None

    if mask is not None:
        mask = mask.bool().npu()

    # PA cache 按 kv_layout 切到 block_size 行 (Bs 维)
    if ENABLE_PA:
        if KV_CACHE_LAYOUT == "BnNBsD":
            k = k[:, :, :BLOCK_SIZE, :]
        elif KV_CACHE_LAYOUT == "BnBsND":
            # BnBsND shape: (Bn, Bs, N, D), Bs 是第 2 维
            k = k[:, :BLOCK_SIZE, :, :]
        elif KV_CACHE_LAYOUT == "NZ":
            # NZ shape: (Bn, N, D/16, Bs, d0), Bs 是倒数第 2 维
            k = k[:, :, :, :BLOCK_SIZE, :]

    logger.info("[NPU] q dtype: %s, shape: %s", q.dtype, q.shape)
    logger.info("[NPU] k dtype: %s, shape: %s", k.dtype, k.shape)
    logger.info(
        "[NPU] dequant_scale_q dtype: %s, shape: %s",
        dequant_scale_q.dtype,
        dequant_scale_q.shape,
    )
    logger.info(
        "[NPU] dequant_scale_k dtype: %s, shape: %s",
        dequant_scale_k.dtype,
        dequant_scale_k.shape,
    )
    logger.info("[NPU] input layout: %s, mask_mode: %s", layout, MASK_MODE)

    atten_out, lse_out = fia_mla_torch_npu(
        q,
        k,
        mask,
        seqused_q,
        cache_seqlens,
        dequant_scale_q,
        dequant_scale_k,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout,
        block_size,
        out_dtype,
    )

    return atten_out, lse_out


def npu_fp8_full_quant_mla(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    seqused_q,
    cache_seqlens,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """主 NPU 量化函数 - 准备数据并调用 NPU

    q_fp8/k_fp8 已为 nope+rope 合并后的 576 维数据并量化
    block_table_torch: 可选外部传入的 block_table（int32 Tensor），用于复现固定场景；
                      None 时根据 NUM_BLOCKS 自动生成。
    返回: (output, cache_info)
        output = (atten_out, lse_out)
        cache_info = (k_pa_clone, v_pa_clone, block_table_np) 或 None（非 PA 或 NUM_BLOCKS==0 时）
    """
    # PA 场景 KV_CACHE_LAYOUT 必须为合法值
    if ENABLE_PA:
        assert KV_CACHE_LAYOUT in {"BnNBsD", "BnBsND", "NZ"}, (
            f"KV_CACHE_LAYOUT must be in {{BnNBsD, BnBsND, NZ}} when ENABLE_PA, got {KV_CACHE_LAYOUT}"
        )

    d_total = D
    softmax_scale = 1.0 / math.sqrt(d_total)
    out_dtype = OUTPUT_DETYPE

    accum_seq_q = (
        make_accum_seq(seqused_q) if INPUT_LAYOUT in ("NTD_TND", "TND") else seqused_q
    )
    npu_input_layout = INPUT_LAYOUT

    q_npu = convert_q_bnsd_to_layout(q_fp8, seqused_q, npu_input_layout)
    deq_q_npu = convert_scale_to_layout(dequant_scale_q, SEQUSED_Q, "deq_q")

    if MASK_MODE == 3:
        mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=q_fp8.device), diagonal=1
        ).npu()
    else:
        mask = None

    cache_info = None
    if ENABLE_PA or not GOLDEN_MODE:
        # block_table: 优先用外部传入的，否则自动生成
        if block_table_torch is not None:
            block_table = block_table_torch.cpu().numpy().astype(np.int32).copy()
            block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)
        else:
            block_table = create_block_table(
                CACHE_SEQLENS, BLOCK_SIZE, num_blocks=NUM_BLOCKS
            )
            block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)

        # K cache (nope+rope 合并的 576 维): 纯数据 block (MLA K per-tensor, deq_k 独立传入)
        k_pa = bnsd_to_k_cache(
            k_fp8,
            CACHE_SEQLENS,
            BLOCK_SIZE,
            block_table,
            num_blocks=NUM_BLOCKS,
            kv_layout=KV_CACHE_LAYOUT,
        )
        # V cache: K 的 nope 部分 (512 维)
        v_pa = bnsd_to_v_cache(
            v_fp8,
            CACHE_SEQLENS,
            BLOCK_SIZE,
            block_table,
            num_blocks=NUM_BLOCKS,
            kv_layout=KV_CACHE_LAYOUT,
        )

        # K_rope 走 PA cache (bf16, 纯数据 block 分片)
        k_rope_pa = None
        if kr_bf16 is not None:
            k_rope_pa = _bnsd_to_pa_bf16(
                kr_bf16,
                CACHE_SEQLENS,
                BLOCK_SIZE,
                block_table,
                num_blocks=NUM_BLOCKS,
                kv_layout=KV_CACHE_LAYOUT,
            )

        if NUM_BLOCKS != 0:
            k_pa_for_golden = k_pa.clone()
            v_pa_for_golden = v_pa.clone()
            block_table_for_golden = block_table.copy()
            k_rope_pa_for_golden = k_rope_pa.clone() if k_rope_pa is not None else None
            cache_info = (
                k_pa_for_golden,
                v_pa_for_golden,
                block_table_for_golden,
                k_rope_pa_for_golden,
            )

        # MLA K per-tensor: deq_k 是标量, 独立传入 (不与 key 共享内存)
        deq_k_npu = dequant_scale_k.float().contiguous()

        if not IS_CONTIGUOUS:
            # torch.stack 不支持 fp8 dtype, 用同宽 uint8 视图 stack 构造非连续 parent 后还原
            kv_cache = torch.stack(
                [k_pa.view(torch.uint8), k_pa.view(torch.uint8)], dim=1
            )
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, 0].view(FP8_DTYPE)
            logger.info(f"[NPU] k_pa is_contiguous={k_pa.is_contiguous()}")
            logger.info(f"[NPU] k_pa strides={k_pa.stride()}")

        output = fa_run_npu(
            q_npu,
            k_pa,
            mask,
            seqused_q,
            cache_seqlens,
            deq_q_npu,
            deq_k_npu,
            p_scale,
            block_table_tensor,
            BLOCK_SIZE,
            N_q,
            N_kv,
            softmax_scale,
            npu_input_layout,
            out_dtype,
        )
    else:
        # 非 PA 模式: K 直接以 BNSD/BSND/TND 输入
        k_npu = convert_q_bnsd_to_layout(k_fp8, cache_seqlens, npu_input_layout)
        # deq_k per-tensor: 标量
        deq_k_npu = convert_scale_to_layout(dequant_scale_k, CACHE_SEQLENS, "deq_v")

        accum_seq_kv = (
            make_accum_seq(cache_seqlens)
            if npu_input_layout in ("TND", "NTD_TND")
            else cache_seqlens
        )

        output = fa_run_npu(
            q_npu,
            k_npu,
            mask,
            accum_seq_q,
            accum_seq_kv,
            deq_q_npu,
            deq_k_npu,
            p_scale,
            None,
            BLOCK_SIZE,
            N_q,
            N_kv,
            softmax_scale,
            npu_input_layout,
            out_dtype,
        )

    atten_out = output[0]
    T_actual = sum(seqused_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]

    return output, cache_info


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    if __package__:
        from . import golden_cache
    else:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="QMLA FullQuant MLA Golden")
    parser.add_argument(
        "--mode",
        default="all",
        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare",
    )
    parser.add_argument(
        "--case-name", default="default", help="case 名称，用于 .pt 文件命名"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="显式启用磁盘缓存；默认直接生成，不保存/加载 tensor",
    )
    parser.add_argument("--gen-device", choices=("npu", "cpu"), default="npu")
    parser.add_argument(
        "--golden-device",
        choices=("npu", "cpu"),
        default="npu",
        help="golden 计算设备；cpu 模式名/缓存文件名保留兼容",
    )
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument(
        "--flush-l2", action="store_true", help="每轮用 256 MiB read 刷 L2"
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1")
    OP_WARMUP, OP_RUNS = args.warmup, args.runs
    OP_FLUSH_L2 = args.flush_l2
    DEVICE_ID = args.device_id
    torch.npu.set_device(DEVICE_ID)
    if args.golden_device == "npu":
        # Keep FP32 reference matmul from silently using reduced precision.
        torch.npu.config.allow_internal_format = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if not raw_parts:
        parser.error("--mode must not be empty")
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    case_name = args.case_name
    cdir = args.cache_dir

    logger.info("=" * 60)
    logger.info("QMLA FullQuant MLA Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("=" * 60)
    logger.info(
        "场景: %s, INPUT_LAYOUT=%s, OUTPUT_LAYOUT=%s",
        "PA" if ENABLE_PA else "noPA",
        INPUT_LAYOUT,
        OUTPUT_LAYOUT,
    )
    logger.info(
        "B=%d, N_q=%d, N_kv=%d, D=%d, D_rope=%d, D_V=%d", B, N_q, N_kv, D, D_rope, D_V
    )
    logger.info("SEQUSED_Q=%s, CACHE_SEQLENS=%s", SEQUSED_Q, CACHE_SEQLENS)

    block_table_torch = None
    if "gen" in mode or cdir is None:
        logger.info("\n[Step 1] 数据生成")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            qr_bf16,
            kr_bf16,
        ) = generate_data(device=args.gen_device)
        if cdir is not None:
            golden_cache.save_input(
                case_name,
                golden_cache.build_input_dict(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    dequant_scale_q,
                    dequant_scale_k,
                    dequant_scale_v,
                    p_scale,
                    qr_bf16,
                    kr_bf16,
                    None,
                    NUM_BLOCKS,
                    KV_CACHE_LAYOUT,
                ),
                cache_dir=cdir,
            )
    else:
        logger.info("[Step 1] 加载已保存的输入数据")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            qr_bf16,
            kr_bf16,
            block_table_torch,
            num_blocks_loaded,
            kv_layout_loaded,
        ) = golden_cache.load_input(case_name, cache_dir=cdir, device=args.gen_device)
        NUM_BLOCKS = num_blocks_loaded
        KV_CACHE_LAYOUT = kv_layout_loaded

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("[Done] 数据生成完成")
        exit(0)

    if "cpu" in mode or ("compare" in mode and cdir is None):
        logger.info("\n[Step 2] Golden (%s)", args.golden_device)
        cpu_out, cpu_lse = fp8_fullquant_mla_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            SEQUSED_Q,
            CACHE_SEQLENS,
            qr_bf16,
            kr_bf16,
            golden_device=args.golden_device,
        )
        if cdir is not None:
            golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    elif "compare" in mode:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(
            case_name, cache_dir=cdir, device=args.golden_device
        )

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("[Done] Golden 计算完成")
        exit(0)

    cache_info = None
    if "npu" in mode or ("compare" in mode and cdir is None):
        logger.info("\n[Step 3] NPU 调用")
        output, cache_info = npu_fp8_full_quant_mla(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            SEQUSED_Q,
            CACHE_SEQLENS,
            block_table_torch,
            qr_bf16,
            kr_bf16,
        )
        atten_out, lse_out = output
        if cdir is not None:
            golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    elif "compare" in mode:
        atten_out, lse_out = golden_cache.load_npu_output(
            case_name, cache_dir=cdir, device="npu"
        )

    if "compare" not in mode:
        logger.info("[Done] NPU 调用完成")
        exit(0)

    # NUM_BLOCKS != 0 时重建 CPU golden
    if cache_info is not None and ("cpu" in mode or "compare" in mode):
        k_pa_cache, v_pa_cache, bt_cache, k_rope_pa_cache = cache_info
        k_bnsd_recon, v_bnsd_recon, kr_bnsd_recon = pa_cache_to_bnsd(
            k_pa_cache,
            v_pa_cache,
            bt_cache,
            CACHE_SEQLENS,
            BLOCK_SIZE,
            kv_layout=KV_CACHE_LAYOUT,
            n_kv=N_kv,
            k_rope_pa=k_rope_pa_cache,
        )
        kr_bf16_recon = (
            kr_bnsd_recon.to(torch.bfloat16) if kr_bnsd_recon is not None else kr_bf16
        )
        cpu_out, cpu_lse = fp8_fullquant_mla_golden(
            q_fp8,
            k_bnsd_recon,
            v_bnsd_recon,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            SEQUSED_Q,
            CACHE_SEQLENS,
            qr_bf16,
            kr_bf16_recon,
            golden_device=args.golden_device,
        )
        if "cpu" in mode and cdir is not None:
            golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    # layout_out 恒等于 INPUT_LAYOUT, PA 与非 PA 均按 INPUT_LAYOUT 对比
    compare_layout = INPUT_LAYOUT
    cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, SEQUSED_Q, compare_layout)
    status, _, _ = result_compare_method.check_result_device(cpu_tnd_torch, atten_out)
    if status != "Pass":
        raise AssertionError("Attention output comparison failed")

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        # LSE输出规格: BSND/BNSD输入 -> BNS[B,N,S]; TND/TND_NTD输入 -> NT
        lse_compare_layout = (
            "BNSD" if INPUT_LAYOUT in ("BSND", "BNSD") else compare_layout
        )
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(
            cpu_lse, SEQUSED_Q, lse_compare_layout
        )
        status, _, _ = result_compare_method.check_result_device(
            cpu_lse_tnd_torch, lse_out
        )
        if status != "Pass":
            raise AssertionError("LSE comparison failed")
