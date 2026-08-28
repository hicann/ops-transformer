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

功能：生成 BNSD 数据 → FP8 per-token-head (Q/K) / per-tensor (KV) 量化 → CPU golden → layout 转换 → NPU 调用 → 精度对比
支持：PA / 非 PA 场景，MLA (DeepSeek 风格: D_nope=512 + D_rope=64, dV=512)
量化：Q per-token-head (quant_mode=3), K/V per-tensor (quant_mode=0)
MLA 特性: bmm1 = Q_nope@K_nope^T + Q_rope@K_rope^T (rope 部分用 bf16 累加), bmm2 = P@V
          softmax_scale = 1/sqrt(D) (仅 nope 维度 D=512, 不含 rope)
"""

import argparse
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng

try:
    from . import result_compare_method
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 3=静态图, 5=动态图, 6=tiling下沉, 7=aclgraph
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 10
N_kv = 1

# MLA dims: D_nope + D_rope = qkHeadDim
D = 512  # nope head dim (Q/K)
D_V = 512  # V head dim
D_rope = 64  # rope head dim

ACTUAL_SEQ_Q = [11]
ACTUAL_SEQ_KV = [570]
# Layout 选择 (TND: query 排布 (T, N_q, D); deq_q 用 TN (T, N_q) 第0维与 query 一致)
INPUT_LAYOUT = "TND"
OUTPUT_LAYOUT = "TND"

# PA KV Cache Layout：BnBsH、BnNBsD、NZ
KV_CACHE_LAYOUT = "BnBsH"

# Data Range (lo, hi)
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-5.0, 5.0)
V_DATA_RANGE = (-5.0, 5.0)

ENABLE_PA = True
ENABLE_LSE = False
GOLDEN_MODE = True
BLOCK_SIZE = 128
SPARSE_MODE = 0
SCALE_VALUE = None
IS_CONTIGUOUS = True

# Q scale layout 由 INPUT_LAYOUT 自动推导, 不再暴露独立的 Q_SCALE_LAYOUT 字段
_Q_SCALE_LAYOUT_MAP = {
    "BNSD": "BNS",
    "BSND": "BSN",
    "TND": "TN",
    "NTD_TND": "NT",
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
def create_block_table(actual_seq_kv, block_size, seed=SEED_BLOCK_TABLE, num_blocks=0):
    """创建 block table，num_blocks 控制物理块复用"""
    block_num_per_batch = [
        math.ceil(int(seq_len) / block_size) for seq_len in actual_seq_kv
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

    block_table = np.full((len(actual_seq_kv), max_blocks), -1, dtype=np.int32)
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
    elif kv_layout == "BnBsH":
        bn, n, bs, d = out_cache.shape
        return out_cache.transpose(1, 2).reshape(bn, bs, n * d).contiguous()
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


# ==============================================================================
# 数据生成
# ==============================================================================
def generate_data():
    """生成 BNSD FP16 Q/K/V (nope) + bf16 Q_rope/K_rope 并做 FP8 量化

    MLA: Q nope 部分走 FP8 per-token-head 量化
         K/V 走 FP8 per-tensor 量化
         Q_rope/K_rope 直接以 bf16 传入 (不量化)
    """
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV) if max(ACTUAL_SEQ_KV) > 0 else 1
    logger.info("[INFO] max_sq=%d, max_skv=%d", max_sq, max_skv)

    def _generate_one(seed, data_range, shape, amp_shape):
        """用 base * amp 结构生成数据，最后线性映射到精确范围 [data_range[0], data_range[1]]"""
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

    q_fp16 = _generate_one(
        SEED_Q, Q_DATA_RANGE, (B, N_q, max_sq, D), (B, N_q, max_sq, 1)
    )
    # MLA: K 和 V 共享同一个 latent tensor (相同数据, 相同 scale)
    # V head dim = K head dim = D = 512
    k_fp16 = _generate_one(
        SEED_K, K_DATA_RANGE, (B, N_kv, max_skv, D), (B, N_kv, max_skv, 1)
    )

    # rope: bf16 不量化 (DeepSeek MLA 风格)
    np.random.seed(SEED_QR)
    qr_bf16 = torch.from_numpy(
        (np.random.uniform(-1.0, 1.0, size=(B, N_q, max_sq, D_rope))).astype(np.float32)
    ).to(torch.bfloat16)
    np.random.seed(SEED_KR)
    kr_bf16 = torch.from_numpy(
        (np.random.uniform(-1.0, 1.0, size=(B, N_kv, max_skv, D_rope))).astype(
            np.float32
        )
    ).to(torch.bfloat16)

    q_fp16 = q_fp16.cpu().contiguous()
    k_fp16 = k_fp16.cpu().contiguous()
    qr_bf16 = qr_bf16.cpu().contiguous()
    kr_bf16 = kr_bf16.cpu().contiguous()

    # MLA 量化: Q per-token-head, K/V per-tensor (K 和 V 共享同一 tensor 和 scale)
    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_tensor_quant_scale(k_fp16)

    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    # MLA: V = K, deq_v = deq_k (相同 scale)
    dequant_scale_v = dequant_scale_k.contiguous()

    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    # MLA: V 使用与 K 相同的 fp8 数据 (指向同一块 tensor)
    v_fp8 = k_fp8.contiguous()

    if max(ACTUAL_SEQ_KV) == 0:
        real_skv = max(ACTUAL_SEQ_KV)
        k_fp8 = k_fp8[:, :, :real_skv, :].contiguous()
        v_fp8 = v_fp8[:, :, :real_skv, :].contiguous()
        kr_bf16 = kr_bf16[:, :, :real_skv, :].contiguous()

    logger.info("[INFO] q_fp8 shape: %s, dtype: %s", q_fp8.shape, q_fp8.dtype)
    logger.info("[INFO] k_fp8 shape: %s, dtype: %s", k_fp8.shape, k_fp8.dtype)
    logger.info("[INFO] v_fp8 shape: %s, dtype: %s", v_fp8.shape, v_fp8.dtype)
    logger.info("[INFO] qr_bf16 shape: %s, dtype: %s", qr_bf16.shape, qr_bf16.dtype)
    logger.info("[INFO] kr_bf16 shape: %s, dtype: %s", kr_bf16.shape, kr_bf16.dtype)
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

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32).cpu().contiguous()

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
def _ntd_to_bnsd(q_ntd, actual_seq_q, n_q):
    """NTD (N_q, T, D) → BNSD (B, N_q, max_Sq, D)"""
    b = len(actual_seq_q)
    max_sq = max(actual_seq_q)
    q_bnsd = torch.zeros((b, n_q, max_sq, q_ntd.shape[-1]), dtype=q_ntd.dtype)
    offset = 0
    for b_idx in range(b):
        act = actual_seq_q[b_idx]
        for n in range(n_q):
            q_bnsd[b_idx, n, :act, :] = q_ntd[n, offset : offset + act, :]
        offset += act
    return q_bnsd


def _bnbd_to_bnsd(kv_bnbd, block_table, actual_seq_kv, block_size):
    """BNBD (block_num, N_kv, block_size, D) → BNSD (B, N_kv, max_Skv, D)"""
    b = len(actual_seq_kv)
    n_kv = kv_bnbd.shape[1]
    d_dim = kv_bnbd.shape[-1]
    max_skv = max(max(actual_seq_kv), 1)
    kv_bnsd = torch.zeros((b, n_kv, max_skv, d_dim), dtype=kv_bnbd.dtype)

    for b_idx in range(b):
        seq_len = actual_seq_kv[b_idx]
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


def _nt_to_bns1(q_scale_nt, actual_seq_q, n_q):
    """NT (N_q, T) → BNS1 (B, N_q, max_Sq, 1)"""
    b = len(actual_seq_q)
    max_sq = max(actual_seq_q)
    q_scale_bns1 = torch.zeros((b, n_q, max_sq, 1), dtype=torch.float32)
    offset = 0
    for b_idx in range(b):
        act = actual_seq_q[b_idx]
        for n in range(n_q):
            q_scale_bns1[b_idx, n, :act, 0] = q_scale_nt[n, offset : offset + act]
        offset += act
    return q_scale_bns1


def _bnb_to_bns1(k_scale_bnb, block_table, actual_seq_kv, block_size):
    """BNB (block_num, N_kv, block_size) → BNS1 (B, N_kv, max_Skv, 1)"""
    b = len(actual_seq_kv)
    n_kv = k_scale_bnb.shape[1]
    max_skv = max(max(actual_seq_kv), 1)
    k_scale_bns1 = torch.zeros((b, n_kv, max_skv, 1), dtype=torch.float32)

    for b_idx in range(b):
        seq_len = actual_seq_kv[b_idx]
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
    BnBsH 还原需要 n_kv 用于拆分 H=N*D
    """
    if kv_layout == "BnNBsD":
        return kv_pa.contiguous()
    elif kv_layout == "BnBsH":
        bn, bs, h = kv_pa.shape
        n = n_kv
        d = h // n
        return kv_pa.reshape(bn, bs, n, d).transpose(1, 2).contiguous()
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
    actual_seq_kv,
    block_size,
    kv_layout="BnNBsD",
    n_kv=None,
    k_rope_pa=None,
):
    """从 PA cache 还原 BNSD 格式的 K/V (纯数据 block), 用于 NUM_BLOCKS 非默认时做 CPU Golden 对比

    MLA K/V per-tensor: deq_k/deq_v 是独立标量, 不在 cache 中
    k_rope_pa: 可选的 rope PA cache (bf16), 同样受物理块复用影响, 需还原
    返回 (k_bnsd, v_bnsd, kr_bnsd)  kr_bnsd 为 None 表示无 rope
    """
    k_bnbd = _pa_layout_to_bnbd(k_pa, kv_layout, n_kv=n_kv)
    v_bnbd = _pa_layout_to_bnbd(v_pa, kv_layout, n_kv=n_kv)
    k_data = k_bnbd[:, :, :block_size, :].contiguous().float()
    v_data = v_bnbd[:, :, :block_size, :].contiguous().float()

    k_bnsd = _bnbd_to_bnsd(k_data, block_table, actual_seq_kv, block_size)
    v_bnsd = _bnbd_to_bnsd(v_data, block_table, actual_seq_kv, block_size)

    kr_bnsd = None
    if k_rope_pa is not None:
        kr_bnbd = _pa_layout_to_bnbd(k_rope_pa, kv_layout, n_kv=n_kv)
        kr_data = kr_bnbd[:, :, :block_size, :].contiguous().float()
        kr_bnsd = _bnbd_to_bnsd(kr_data, block_table, actual_seq_kv, block_size)

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


def cpu_fp8_fullquant_mla_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    deq_q,
    deq_k,
    deq_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    qr_bf16=None,
    kr_bf16=None,
):
    """CPU golden reference - MLA flash attention with FP8 quantization

    Q/K: fp8 per-token-head quant, dequant before matmul
    V:   fp8 per-tensor quant
    Q_rope/K_rope: bf16, 不量化, 直接参与 bmm1 累加
    softmax_scale = 1/sqrt(D) (仅 nope 部分, 不含 rope)
    """
    softmax_scale = get_softmax_scale(SCALE_VALUE, D)

    q_tensor = q_fp8.cpu().to(torch.float32).contiguous()
    k_tensor = k_fp8.cpu().to(torch.float32).contiguous()
    v_tensor = v_fp8.cpu().to(torch.float32).contiguous()
    deq_q = deq_q.cpu().float().contiguous()
    deq_k = deq_k.cpu().float().contiguous()
    deq_v = deq_v.cpu().float().contiguous()

    # GQA-style broadcast (N_q > N_kv): 仅 K/V 数据和 rope 需要 broadcast
    # K/V per-tensor: deq_k/deq_v 是标量, 不需要 broadcast
    if N_q != N_kv:
        k_tensor = torch_broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = torch_broadcast_kv(N_q, N_kv, v_tensor)
        if kr_bf16 is not None:
            kr_bf16 = torch_broadcast_kv(N_q, N_kv, kr_bf16)

    batch, heads, q_seq, d_dim = q_tensor.shape
    v_dim = v_tensor.shape[-1]

    # 空 KV 场景
    if k_tensor.shape[2] == 0:
        result = torch.zeros(
            (batch, heads, q_seq, v_dim), dtype=torch.float32
        ).contiguous()
        lse = torch.full(
            (batch, heads, q_seq, 1), float("inf"), dtype=torch.float32
        ).contiguous()
        return result, lse

    out = torch.zeros((batch, heads, q_seq, v_dim), dtype=torch.float32).contiguous()
    o_sum = torch.zeros(q_tensor.shape[:-1], dtype=torch.float32)[
        ..., None
    ].contiguous()
    minValue = torch.tensor(-3.402823466e38, dtype=torch.float32)
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[
        ..., None
    ].contiguous()

    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32).contiguous()
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32).contiguous()
    q_lens_acl = q_lens_t.view(batch, 1, 1, 1).contiguous()
    k_lens_acl = k_lens_t.view(batch, 1, 1, 1).contiguous()

    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]
    q_range = torch.arange(Sq).view(1, 1, -1, 1).contiguous()
    k_range = torch.arange(Skv).view(1, 1, 1, -1).contiguous()
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if SPARSE_MODE == 3:
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

    # rope blocks (bf16)
    qr_blocks = None
    kr_blocks = None
    if qr_bf16 is not None and kr_bf16 is not None:
        qr_blocks = list(torch.split(qr_bf16, Q_BLOCK_SIZE, dim=2))
        kr_blocks = list(torch.split(kr_bf16, KV_BLOCK_SIZE, dim=2))

    # K/V per-tensor: deq_k/deq_v 是标量
    deq_k_scalar = deq_k.reshape(-1)[0] if deq_k.numel() == 1 else deq_k
    deq_v_scalar = deq_v.reshape(-1)[0] if deq_v.numel() == 1 else deq_v
    fp8_e4m3_max = 448.0

    for j, (kj, vj) in enumerate(zip(k_blocks, v_blocks)):
        kj = kj.contiguous()
        kj_T = kj.transpose(-1, -2).contiguous()
        vj = vj.contiguous()

        krj = kr_blocks[j] if kr_blocks is not None else None
        krj_T = krj.transpose(-1, -2).contiguous() if krj is not None else None

        for i, qi in enumerate(q_blocks):
            oi = o_blocks[i]
            si = s_blocks[i]
            mi = m_blocks[i]
            deq_qi = deq_q_blocks[i]
            qri = qr_blocks[i] if qr_blocks is not None else None

            # bmm1: (Q_nope@K_nope^T + Q_rope@K_rope^T) * deq_q * deq_k * scalar
            # rope 与 nope 在 L0C 累加后, 整体乘 deq_q * deq_k * scalar (与 NPU 一致)
            sij = torch.matmul(qi, kj_T)
            if qri is not None and krj_T is not None:
                rope_score = torch.matmul(
                    qri.to(torch.float32), krj_T.to(torch.float32)
                )
                sij = sij + rope_score
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

            # MLA rescale: quantScale = 448 / rowMax_A (与 NPU isMlaFullQuant=true 一致)
            rowMax_A = torch.exp(m_block - mi_new) + EPSILON
            quantScale = fp8_e4m3_max / rowMax_A
            pij_scaled = (pij * quantScale).to(FP8_DTYPE).to(torch.float32)

            # bmm2: P_fp8 @ V, 乘 rowMax_A (deq_v 在最终输出时统一乘)
            bmm2_res = torch.matmul(pij_scaled, vj)
            bmm2_res = bmm2_res * rowMax_A

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
    tensor = tensor.cpu().contiguous()
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
    tensor = tensor.cpu().contiguous()
    if scale_type == "deq_q":
        b, n, _, _ = tensor.shape
        T = sum(seq_lens)
        # Q scale layout 由 INPUT_LAYOUT 自动推导 (不暴露 Q_SCALE_LAYOUT 字段)
        q_scale_layout = _Q_SCALE_LAYOUT_MAP.get(INPUT_LAYOUT, "BNS")
        if q_scale_layout == "NT":
            result = torch.zeros((n, T), dtype=torch.float32)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[n_idx, t : t + act_s] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif q_scale_layout == "TN":
            result = torch.zeros((T, n), dtype=torch.float32)
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
            result = torch.zeros((b, n, s_dim), dtype=torch.float32)
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[b_idx, n_idx, :act_s] = tensor[b_idx, n_idx, :act_s, 0]
            return result.contiguous()
        elif q_scale_layout == "BSN":
            s_dim = tensor.shape[2]
            result = torch.zeros((b, s_dim, n), dtype=torch.float32)
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
    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        q_rope,
        k_rope,
        mask,
        actual_seq_q,
        actual_seq_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout,
        block_size,
        out_dtype,
    ):
        atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
            q,
            k,
            v,
            query_rope=q_rope,
            key_rope=k_rope,
            atten_mask=mask,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            dequant_scale_query=dequant_scale_q,
            dequant_scale_key=dequant_scale_k,
            dequant_scale_value=dequant_scale_v,
            block_table=block_table,
            block_size=block_size,
            num_query_heads=q_n,
            num_key_value_heads=kv_n,
            softmax_scale=softmax_scale,
            input_layout=layout,
            sparse_mode=SPARSE_MODE,
            out_dtype=out_dtype,
            **get_npu_fa_kwargs(),
        )
        return atten_out, lse_out


def call_npu_fa_op(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    block_size,
    out_dtype,
):
    torch.npu.synchronize()
    atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        query_rope=q_rope,
        key_rope=k_rope,
        atten_mask=mask,
        actual_seq_qlen=actual_seq_q,
        actual_seq_kvlen=actual_seq_kv,
        dequant_scale_query=dequant_scale_q,
        dequant_scale_key=dequant_scale_k,
        dequant_scale_value=dequant_scale_v,
        block_table=block_table,
        block_size=block_size,
        num_query_heads=q_n,
        num_key_value_heads=kv_n,
        softmax_scale=softmax_scale,
        input_layout=layout,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
        **get_npu_fa_kwargs(),
    )
    torch.npu.synchronize()
    return atten_out, lse_out


def fia_mla_torch_npu(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
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
            v,
            q_rope,
            k_rope,
            mask,
            actual_seq_q,
            actual_seq_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout,
            block_size,
            out_dtype,
        )

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    config = CompilerConfig()
    with torch.no_grad():
        torch.npu.synchronize()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        fa_args = (
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            actual_seq_q,
            actual_seq_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout,
            block_size,
            out_dtype,
        )

        if GRAPH_PATH == 3:
            logger.info("[NPU] GRAPH_PATH == 3, 静态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=False
            )
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH == 5:
            logger.info("[NPU] GRAPH_PATH == 5, 动态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH in (6, 7):
            # tiling 下沉 / aclgraph: 需要标记静态 shape
            if GRAPH_PATH == 7:
                logger.info("[NPU] GRAPH_PATH == 7, aclgraph...")
                config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
                config.mode = "reduce-overhead"
            else:
                logger.info("[NPU] GRAPH_PATH == 6, tiling下沉...")
                config.experimental_config.tiling_schedule_optimize = True
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            for t in (
                q,
                k,
                v,
                q_rope,
                k_rope,
                mask,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                block_table,
            ):
                if t is not None:
                    torch._dynamo.mark_static(t)
            atten_out, lse_out = npu_mode(*fa_args)
        else:
            raise ValueError(
                f"Unsupported GRAPH_PATH: {GRAPH_PATH}, only support 0/3/5/6/7"
            )

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out


def fa_run_npu(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
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

    # 非 TND 场景 (BNSD/BSND) 不能传 actual_seq 给算子, 由 tensor shape 推断 S
    if layout not in ("TND", "NTD_TND"):
        actual_seq_q = None
        actual_seq_kv = None
    if ENABLE_PA:
        actual_seq_kv = ACTUAL_SEQ_KV

    q = q.npu()
    k = k.npu()
    v = v.npu()
    q_rope = q_rope.npu() if q_rope is not None else None
    k_rope = k_rope.npu() if k_rope is not None else None

    # MLA K per-tensor: deq_k 独立传入 (标量, 不与 key 共享内存)
    dequant_scale_k = dequant_scale_k.float().npu()

    dequant_scale_q = dequant_scale_q.float().npu()
    dequant_scale_v = dequant_scale_v.float().npu()
    p_scale = p_scale.float().npu()

    block_table = block_table.int().npu() if ENABLE_PA else None

    if mask is not None:
        mask = mask.bool().npu()

    # PA cache 按 kv_layout 切到 block_size 行 (Bs 维)
    if ENABLE_PA:
        if KV_CACHE_LAYOUT == "BnNBsD":
            k = k[:, :, :BLOCK_SIZE, :]
            v = v[:, :, :BLOCK_SIZE, :]
        elif KV_CACHE_LAYOUT == "BnBsH":
            # BnBsH shape: (Bn, Bs, H), H = N_kv * D, Bs 是倒数第 2 维
            k = k[:, :BLOCK_SIZE, :]
            v = v[:, :BLOCK_SIZE, :]
        elif KV_CACHE_LAYOUT == "NZ":
            # NZ shape: (Bn, N, D/16, Bs, d0), Bs 是倒数第 2 维
            k = k[:, :, :, :BLOCK_SIZE, :]
            v = v[:, :, :, :BLOCK_SIZE, :]
        # rope 不切片 (与 MX _prepare_rope_npu 保持一致)

    logger.info("[NPU] q dtype: %s, shape: %s", q.dtype, q.shape)
    logger.info("[NPU] k dtype: %s, shape: %s", k.dtype, k.shape)
    logger.info("[NPU] v dtype: %s, shape: %s", v.dtype, v.shape)
    logger.info(
        "[NPU] q_rope dtype: %s, shape: %s",
        q_rope.dtype if q_rope is not None else None,
        q_rope.shape if q_rope is not None else None,
    )
    logger.info(
        "[NPU] k_rope dtype: %s, shape: %s",
        k_rope.dtype if k_rope is not None else None,
        k_rope.shape if k_rope is not None else None,
    )
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
    logger.info(
        "[NPU] dequant_scale_v dtype: %s, shape: %s",
        dequant_scale_v.dtype,
        dequant_scale_v.shape,
    )
    logger.info("[NPU] input layout: %s, sparse_mode: %s", layout, SPARSE_MODE)

    atten_out, lse_out = fia_mla_torch_npu(
        q,
        k,
        v,
        q_rope,
        k_rope,
        mask,
        actual_seq_q,
        actual_seq_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout,
        block_size,
        out_dtype,
    )

    if GRAPH_PATH == 0:
        atten_out = atten_out.cpu()
        lse_out = lse_out.cpu()
    return atten_out, lse_out


def npu_fp8_full_quant_mla(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """主 NPU 量化函数 - 准备数据并调用 NPU

    block_table_torch: 可选外部传入的 block_table（int32 Tensor），用于复现固定场景；
                      None 时根据 NUM_BLOCKS 自动生成。
    返回: (output, cache_info)
        output = (atten_out, lse_out)
        cache_info = (k_pa_clone, v_pa_clone, block_table_np) 或 None（非 PA 或 NUM_BLOCKS==0 时）
    """
    # PA 场景 KV_CACHE_LAYOUT 必须为合法值
    if ENABLE_PA:
        assert KV_CACHE_LAYOUT in {"BnNBsD", "BnBsH", "NZ"}, (
            f"KV_CACHE_LAYOUT must be in {{BnNBsD, BnBsH, NZ}} when ENABLE_PA, got {KV_CACHE_LAYOUT}"
        )

    d_total = D
    softmax_scale = 1.0 / math.sqrt(d_total)
    out_dtype = OUTPUT_DETYPE

    accum_seq_q = (
        make_accum_seq(actual_seq_q)
        if INPUT_LAYOUT in ("NTD_TND", "TND")
        else actual_seq_q
    )
    npu_input_layout = INPUT_LAYOUT

    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, npu_input_layout)
    deq_q_npu = convert_scale_to_layout(dequant_scale_q, ACTUAL_SEQ_Q, "deq_q")
    deq_v_npu = convert_scale_to_layout(dequant_scale_v, ACTUAL_SEQ_KV, "deq_v")

    # rope layout 转换
    # Q_rope 直接传原始 bf16 (不预量化)
    # NPU: nope 部分乘 descaleQK=deq_q*deq_k, rope 部分不经过 descaleQK
    q_rope_npu = None
    if qr_bf16 is not None:
        q_rope_npu = convert_q_bnsd_to_layout(qr_bf16, actual_seq_q, npu_input_layout)
    k_rope_npu = None
    if kr_bf16 is not None:
        k_rope_npu = convert_q_bnsd_to_layout(kr_bf16, actual_seq_kv, npu_input_layout)

    if SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
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
                ACTUAL_SEQ_KV, BLOCK_SIZE, num_blocks=NUM_BLOCKS
            )
            block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)

        # K cache (nope): 纯数据 block (MLA K per-tensor, deq_k 独立传入)
        k_pa = bnsd_to_k_cache(
            k_fp8,
            ACTUAL_SEQ_KV,
            BLOCK_SIZE,
            block_table,
            num_blocks=NUM_BLOCKS,
            kv_layout=KV_CACHE_LAYOUT,
        )
        # V cache: 纯数据 block
        v_pa = bnsd_to_v_cache(
            v_fp8,
            ACTUAL_SEQ_KV,
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
                ACTUAL_SEQ_KV,
                BLOCK_SIZE,
                block_table,
                num_blocks=NUM_BLOCKS,
                kv_layout=KV_CACHE_LAYOUT,
            )
        if not IS_CONTIGUOUS:
            # ---- 构造krope不连续 ----
            fake_krope_tensor = torch.ones_like(k_rope_pa)
            double_krope = torch.stack([k_rope_pa, fake_krope_tensor], dim=1)
            double_krope = double_krope.npu()
            k_rope_pa = double_krope[:, 0]  # 覆写为非连续
            logger.info(f"[NPU] k_rope is_contiguous={k_rope_pa.is_contiguous()}")
            logger.info(f"[NPU] k_rope strides={k_rope_pa.stride()}")

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
            kv_cache = torch.stack([k_pa, v_pa], dim=1)
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, 0]
            v_pa = kv_cache[:, 1]
            logger.info(
                f"[NPU] k_pa is_contiguous={k_pa.is_contiguous()}, v_pa is_contiguous={v_pa.is_contiguous()}"
            )
            logger.info(
                f"[NPU] k_pa strides={k_pa.stride()}, v_pa strides={v_pa.stride()}"
            )

        output = fa_run_npu(
            q_npu,
            k_pa,
            v_pa,
            q_rope_npu,
            k_rope_pa,
            mask,
            accum_seq_q,
            actual_seq_kv,
            deq_q_npu,
            deq_k_npu,
            deq_v_npu,
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
        # 非 PA 模式: K/V 直接以 BNSD/BSND/TND 输入
        k_npu = convert_q_bnsd_to_layout(k_fp8, actual_seq_kv, npu_input_layout)
        v_npu = convert_q_bnsd_to_layout(v_fp8, actual_seq_kv, npu_input_layout)
        # deq_k per-tensor: 标量, 与 deq_v 相同处理
        deq_k_npu = convert_scale_to_layout(dequant_scale_k, ACTUAL_SEQ_KV, "deq_v")

        accum_seq_kv = (
            make_accum_seq(actual_seq_kv)
            if npu_input_layout in ("TND", "NTD_TND")
            else actual_seq_kv
        )

        output = fa_run_npu(
            q_npu,
            k_npu,
            v_npu,
            q_rope_npu,
            k_rope_npu,
            mask,
            accum_seq_q,
            accum_seq_kv,
            deq_q_npu,
            deq_k_npu,
            deq_v_npu,
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
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]

    return output, cache_info


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
# Main
# ==============================================================================
if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="FIA FullQuant MLA Golden")
    parser.add_argument(
        "--mode",
        default="all",
        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare",
    )
    parser.add_argument(
        "--case-name", default="default", help="case 名称，用于 .pt 文件命名"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="缓存目录路径（默认 golden_cache/）"
    )
    args = parser.parse_args()

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    case_name = args.case_name
    cdir = args.cache_dir

    logger.info("=" * 60)
    logger.info("FIA FullQuant MLA Golden  [mode=%s, case=%s]", mode, case_name)
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
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)

    block_table_torch = None
    if "gen" in mode:
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
        ) = generate_data()
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
        ) = golden_cache.load_input(case_name, cache_dir=cdir)
        NUM_BLOCKS = num_blocks_loaded
        KV_CACHE_LAYOUT = kv_layout_loaded

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("[Done] 数据已保存，退出")
        exit(0)

    if "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        cpu_out, cpu_lse = cpu_fp8_fullquant_mla_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            ACTUAL_SEQ_Q,
            ACTUAL_SEQ_KV,
            qr_bf16,
            kr_bf16,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("[Done] CPU 输出已保存，退出")
        exit(0)

    cache_info = None
    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        output, cache_info = npu_fp8_full_quant_mla(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            ACTUAL_SEQ_Q,
            ACTUAL_SEQ_KV,
            block_table_torch,
            qr_bf16,
            kr_bf16,
        )
        atten_out, lse_out = output
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("[Done] NPU 输出已保存，退出")
        exit(0)

    # NUM_BLOCKS != 0 时重建 CPU golden
    if cache_info is not None and ("cpu" in mode or "compare" in mode):
        k_pa_cache, v_pa_cache, bt_cache, k_rope_pa_cache = cache_info
        k_bnsd_recon, v_bnsd_recon, kr_bnsd_recon = pa_cache_to_bnsd(
            k_pa_cache,
            v_pa_cache,
            bt_cache,
            ACTUAL_SEQ_KV,
            BLOCK_SIZE,
            kv_layout=KV_CACHE_LAYOUT,
            n_kv=N_kv,
            k_rope_pa=k_rope_pa_cache,
        )
        kr_bf16_recon = (
            kr_bnsd_recon.to(torch.bfloat16) if kr_bnsd_recon is not None else kr_bf16
        )
        cpu_out, cpu_lse = cpu_fp8_fullquant_mla_golden(
            q_fp8,
            k_bnsd_recon,
            v_bnsd_recon,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            ACTUAL_SEQ_Q,
            ACTUAL_SEQ_KV,
            qr_bf16,
            kr_bf16_recon,
        )
        if "cpu" in mode:
            golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    compare_layout = "TND" if ENABLE_PA else INPUT_LAYOUT
    cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, ACTUAL_SEQ_Q, compare_layout)
    result_compare_method.check_result(cpu_tnd_torch, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(
            cpu_lse, ACTUAL_SEQ_Q, compare_layout
        )
        result_compare_method.check_result(cpu_lse_tnd_torch, lse_out)
