#!/usr/bin/python
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
FIA FullQuant GQA Golden

功能：生成 BNSD 数据 → FP8 per-token-head / per-head 量化 → CPU golden → layout 转换 → NPU 调用 → 精度对比
支持：PA / 非 PA 场景，GQA，block_table 参数化（NUM_BLOCKS 物理块复用），外部 NPU 排布 pt 加载
量化：Q/K per-token-head (quant_mode=3), V per-head (quant_mode=2)
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
    from . import load_external_data
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method
    import load_external_data

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 16
N_kv = 2
D = 128

ACTUAL_SEQ_Q = [2048]
ACTUAL_SEQ_KV = [2048]

# Layout 选择
INPUT_LAYOUT = "NTD_TND"
OUTPUT_LAYOUT = "TND"
Q_SCALE_LAYOUT = "NT"

# PA KV Cache Layout
KV_CACHE_LAYOUT = "BnNBsD"

# Data Range (lo, hi)
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-1.0, 1.0)
V_DATA_RANGE = (-1.0, 1.0)

ENABLE_PA = True
ENABLE_LSE = True
GOLDEN_MODE = True
BLOCK_SIZE = 128
SPARSE_MODE = 3
SCALE_VALUE = None
IS_CONTIGUOUS = True

# Seed
SEED_Q = 54
SEED_K = 3
SEED_V = 20
SEED_BLOCK_TABLE = 1234
FP8_DTYPE = torch.float8_e4m3fn
OUTPUT_DETYPE = torch.bfloat16
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

# 物理 block 数量，0 表示使用默认值（等于 total_blocks）
NUM_BLOCKS = 0


# ==============================================================================
# 数据生成函数
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


def get_fp8_per_head_quant_scale(tensor):
    """per-head quant scale: shape (1, N, 1, 1)"""
    tensor = tensor.contiguous()
    fp8_e4m3_max = 448.0
    head_max = torch.abs(tensor).amax(dim=(0, 2, 3), keepdim=True)
    head_max = torch.max(head_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / head_max
    return scale.float().contiguous()


def quant_fp16_to_fp8(tensor, scale):
    """将 fp16 数据量化为 fp8_e4m3"""
    tensor = tensor.contiguous()
    scale = scale.contiguous()
    result = tensor.float() * scale
    result = torch.clamp(result, -448.0, 448.0)
    return result.to(FP8_DTYPE).contiguous()


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


def bnsd_to_k_cache(
    k_fp8_bnsd, k_scale_fp32_bnsd, seq_lens, block_size, block_table, num_blocks=0
):
    """BNSD to PA K cache, with k scale (fp32) stored in the 4 extra rows"""
    k_fp8_bnsd = k_fp8_bnsd.contiguous()
    k_scale_fp32_bnsd = k_scale_fp32_bnsd.contiguous()
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape
    scale_rows = 4
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    cache = torch.zeros(
        (cache_blocks, N_dim, block_size + scale_rows, D_dim),
        dtype=torch.uint8,
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
            k_data = k_fp8_bnsd[b, :, start_s:end_s, :].contiguous()
            cache[blockid, :, :valid, :] = k_data.view(torch.uint8)
            scales_all = k_scale_fp32_bnsd[b, :, start_s:end_s, 0].contiguous()
            scale_buf = torch.zeros(
                N_dim, scale_rows, D_dim // 4, dtype=torch.float32, device=cache.device
            )
            flat_scale = scale_buf.reshape(N_dim, -1)
            if valid <= flat_scale.shape[1]:
                flat_scale[:, :valid] = scales_all
            cache[blockid, :, block_size : block_size + scale_rows, :] = scale_buf.view(
                torch.uint8
            ).reshape(N_dim, scale_rows, D_dim)

    return (
        cache.view(FP8_DTYPE)
        .reshape(cache_blocks, N_dim, block_size + scale_rows, D_dim)
        .contiguous()
    )


def bnsd_to_v_cache(tensor_bnsd, seq_lens, block_size, block_table, num_blocks=0):
    """BNSD to V cache - V cache 使用 FP8 类型"""
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    out_cache = torch.zeros(
        (cache_blocks, heads, block_size + 4, dim), dtype=FP8_DTYPE, device=device
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

    return out_cache.contiguous()


def generate_data():
    """生成 BNSD FP16 Q/K/V 并做 FP8 量化"""
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
    k_fp16 = _generate_one(
        SEED_K, K_DATA_RANGE, (B, N_kv, max_skv, D), (B, N_kv, max_skv, 1)
    )
    v_fp16 = _generate_one(SEED_V, V_DATA_RANGE, (B, N_kv, max_skv, D), (B, N_kv, 1, 1))

    q_fp16 = q_fp16.cpu().contiguous()
    k_fp16 = k_fp16.cpu().contiguous()
    v_fp16 = v_fp16.cpu().contiguous()

    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_token_head_quant_scale(k_fp16)
    quant_scale_v = get_fp8_per_head_quant_scale(v_fp16)

    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    dequant_scale_v = (1.0 / quant_scale_v).contiguous()

    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    v_fp8 = quant_fp16_to_fp8(v_fp16, quant_scale_v)

    if max(ACTUAL_SEQ_KV) == 0:
        real_skv = max(ACTUAL_SEQ_KV)
        k_fp8 = k_fp8[:, :, :real_skv, :].contiguous()
        v_fp8 = v_fp8[:, :, :real_skv, :].contiguous()

    logger.info("[INFO] q_fp8 shape: %s, dtype: %s", q_fp8.shape, q_fp8.dtype)
    logger.info("[INFO] k_fp8 shape: %s, dtype: %s", k_fp8.shape, k_fp8.dtype)
    logger.info("[INFO] v_fp8 shape: %s, dtype: %s", v_fp8.shape, v_fp8.dtype)

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32).cpu().contiguous()

    return (
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
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


def pa_cache_to_bnsd(k_pa, v_pa, block_table, actual_seq_kv, block_size):
    """从 PA cache 还原 BNSD 格式的 K/V/deq_k，用于 NUM_BLOCKS 非默认时做 CPU Golden 对比

    返回 (k_bnsd, v_bnsd, deq_k_bns1)
    """
    k_data = k_pa[:, :, :block_size, :].contiguous().float()
    v_data = v_pa[:, :, :block_size, :].contiguous().float()

    k_pa_f32 = (
        k_pa.view(torch.uint8)
        .view(k_pa.shape[0], k_pa.shape[1], -1)
        .view(torch.float32)
    )
    deq_k_flat = k_pa_f32[:, :, -block_size:].contiguous()

    k_bnsd = _bnbd_to_bnsd(k_data, block_table, actual_seq_kv, block_size)
    v_bnsd = _bnbd_to_bnsd(v_data, block_table, actual_seq_kv, block_size)
    deq_k_bns1 = _bnb_to_bns1(deq_k_flat, block_table, actual_seq_kv, block_size)

    return k_bnsd, v_bnsd, deq_k_bns1


def external_to_bnsd(ext):
    """将外部 NPU 排布 tensor 转换为 BNSD 格式，供 CPU Golden 使用

    返回 (q_fp8_bnsd, k_fp8_bnsd, v_fp8_bnsd,
          deq_q_bns1, deq_k_bns1, deq_v_1N11, p_scale)
    """
    logger.info("Converting external (NPU layout) → BNSD for CPU golden...")

    q_bnsd = _ntd_to_bnsd(ext.q_fp8, ACTUAL_SEQ_Q, N_q)
    k_bnsd = _bnbd_to_bnsd(ext.k_fp8, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)
    v_bnsd = _bnbd_to_bnsd(ext.v_fp8, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)

    deq_q_bns1 = _nt_to_bns1(ext.deq_q, ACTUAL_SEQ_Q, N_q)
    deq_k_bns1 = _bnb_to_bns1(ext.deq_k, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)
    deq_v_1N11 = ext.deq_v.reshape(1, N_kv, 1, 1).float()

    logger.info("  q_bnsd: %s", q_bnsd.shape)
    logger.info("  k_bnsd: %s", k_bnsd.shape)
    logger.info("  v_bnsd: %s", v_bnsd.shape)
    logger.info("  deq_q_bns1: %s", deq_q_bns1.shape)
    logger.info("  deq_k_bns1: %s", deq_k_bns1.shape)
    logger.info("  deq_v_1N11: %s", deq_v_1N11.shape)

    return (q_bnsd, k_bnsd, v_bnsd, deq_q_bns1, deq_k_bns1, deq_v_1N11, ext.p_scale)


# ==============================================================================
# CPU Golden 函数
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


def cpu_fp8_fullquant_gqa_golden(
    q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, actual_seq_q, actual_seq_kv
):
    """CPU golden reference - 所有操作在CPU上执行"""
    softmax_scale = get_softmax_scale(SCALE_VALUE, D)
    q_tensor = q_fp8.cpu().to(torch.float32).contiguous()
    batch, heads, q_seq, d_dim = q_tensor.shape

    k_tensor = k_fp8.cpu().to(torch.float32).contiguous()
    v_tensor = v_fp8.cpu().to(torch.float32).contiguous()
    deq_q = deq_q.cpu().float().contiguous()
    deq_k = deq_k.cpu().float().contiguous()
    deq_v = deq_v.cpu().float().contiguous()

    if N_q != N_kv:
        k_tensor = torch_broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = torch_broadcast_kv(N_q, N_kv, v_tensor)
        deq_k = torch_broadcast_kv(N_q, N_kv, deq_k)
        deq_v = torch_broadcast_kv(N_q, N_kv, deq_v)

    batch, heads, q_seq, _ = q_tensor.shape
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
    deq_k_blocks = list(torch.split(deq_k, KV_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor(
        [math.log(p_scale.item())], dtype=torch.float32
    ).contiguous()

    for j, (kj, vj) in enumerate(zip(k_blocks, v_blocks)):
        kj = kj.contiguous()
        kj_T = kj.transpose(-1, -2).contiguous()
        vj = vj.contiguous()
        deq_kj = deq_k_blocks[j]
        deq_kj_T = deq_kj.transpose(-1, -2).contiguous()

        for i, qi in enumerate(q_blocks):
            oi = o_blocks[i]
            si = s_blocks[i]
            mi = m_blocks[i]
            deq_qi = deq_q_blocks[i]

            sij = torch.matmul(qi, kj_T)
            deq_qi = deq_qi * softmax_scale
            sij = sij * deq_qi * deq_kj_T

            causal_mask = mask_blocks[i][j].contiguous()
            sij = sij.masked_fill(causal_mask, float("-inf"))

            m_block, _ = torch.max(sij, dim=-1, keepdims=True)
            m_block = m_block - ln_p_scale
            mi_new = torch.maximum(m_block, mi)
            all_masked_block = m_block == float("-inf")
            pij = torch.where(
                all_masked_block, torch.zeros_like(sij), torch.exp(sij - mi_new)
            )
            s_block = torch.sum(pij, dim=-1, keepdims=True)
            pij_drop = pij.to(FP8_DTYPE).to(torch.float32)
            pij_v = torch.matmul(pij_drop, vj)
            pij_v = pij_v * deq_v
            scale = torch.where(
                mi_new == float("-inf"), torch.ones_like(mi_new), torch.exp(mi - mi_new)
            )
            si_new = scale * si + s_block
            o_blocks[i] = (si * torch.exp(mi - mi_new) * oi + pij_v) / (
                si_new + EPSILON
            )
            s_blocks[i] = si_new
            m_blocks[i] = mi_new

    result = torch.cat(o_blocks, dim=2).contiguous()
    out_sum = torch.cat(s_blocks, dim=2).contiguous()
    out_max = torch.cat(m_blocks, dim=2).contiguous()

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
        if Q_SCALE_LAYOUT == "NT":
            result = torch.zeros((n, T), dtype=torch.float32)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[n_idx, t : t + act_s] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "TN":
            result = torch.zeros((T, n), dtype=torch.float32)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[t : t + act_s, n_idx] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "BNSD":
            result = torch.zeros((b, n, max(seq_lens), 1), dtype=torch.float32)
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[b_idx, n_idx, :act_s, 0] = tensor[b_idx, n_idx, :act_s, 0]
            return result.contiguous()
        else:
            return tensor.float().contiguous()
    elif scale_type == "deq_v":
        return tensor.reshape(tensor.shape[1]).float().contiguous()
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
        query_quant_mode=3,
        key_quant_mode=3,
        value_quant_mode=2,
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
            quant_scale_p=p_scale,
            out_dtype=out_dtype,
            **get_npu_fa_kwargs(),
        )
        return atten_out, lse_out


def call_npu_fa_op(
    q,
    k,
    v,
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
        quant_scale_p=p_scale,
        out_dtype=out_dtype,
        **get_npu_fa_kwargs(),
    )
    torch.npu.synchronize()
    return atten_out, lse_out


def fia_gqa_torch_npu(
    q,
    k,
    v,
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

        if GRAPH_PATH == 5:
            logger.info("[NPU] GRAPH_PATH == 5, 动态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH == 7:
            logger.info("[NPU] GRAPH_PATH == 7, aclgraph...")
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            config.mode = "reduce-overhead"
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            for t in (
                q,
                k,
                v,
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
                f"Unsupported GRAPH_PATH: {GRAPH_PATH}, only support 0/5/7"
            )

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out


def fa_run_npu(
    q,
    k,
    v,
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

    q = q.npu()
    k = k.npu()
    v = v.npu()

    # 从 K cache 的 scale rows 中提取 deq_k（共享内存）
    k_pa_f32 = k.view(torch.uint8).view(k.shape[0], k.shape[1], -1).view(torch.float32)
    dequant_scale_k = k_pa_f32[:, :, -BLOCK_SIZE:]

    dequant_scale_q = dequant_scale_q.float().npu()
    dequant_scale_v = dequant_scale_v.float().npu()
    p_scale = p_scale.float().npu()

    if not IS_CONTIGUOUS and ENABLE_PA:
        fake_kscale_tensor = torch.ones_like(dequant_scale_k)
        double_kscale = torch.stack([dequant_scale_k, fake_kscale_tensor], dim=2)
        double_kscale = double_kscale.npu()
        dequant_scale_k = double_kscale[:, :, 0]

    block_table = block_table.int().npu() if ENABLE_PA else None

    if mask is not None:
        mask = mask.bool().npu()

    # 从 cache 中取数据切片
    k = k[:, :, :128, :]
    v = v[:, :, :128, :]

    logger.info("[NPU] q dtype: %s, shape: %s", q.dtype, q.shape)
    logger.info("[NPU] k dtype: %s, shape: %s", k.dtype, k.shape)
    logger.info("[NPU] v dtype: %s, shape: %s", v.dtype, v.shape)
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
    logger.info(
        "[NPU] key is_contiguous: %s, value is_contiguous: %s",
        k.is_contiguous(),
        v.is_contiguous(),
    )
    logger.info(
        "[NPU] dequant_scale_k is_contiguous: %s", dequant_scale_k.is_contiguous()
    )
    logger.info("[NPU] k stride: %s", k.stride())
    logger.info("[NPU] v stride: %s", v.stride())
    logger.info("[NPU] dequant_scale_k stride: %s", dequant_scale_k.stride())

    atten_out, lse_out = fia_gqa_torch_npu(
        q,
        k,
        v,
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


def npu_fp8_full_quant(
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
):
    """主 NPU 量化函数 - 准备数据并调用 NPU

    block_table_torch: 可选外部传入的 block_table（int32 Tensor），用于复现固定场景；
                      None 时根据 NUM_BLOCKS 自动生成。
    返回: (output, cache_info)
        output = (atten_out, lse_out)
        cache_info = (k_pa_clone, v_pa_clone, block_table_np) 或 None（NUM_BLOCKS==0 时）
    """
    softmax_scale = 1.0 / math.sqrt(D)
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

    if SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None

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

        k_pa = bnsd_to_k_cache(
            k_fp8,
            dequant_scale_k,
            ACTUAL_SEQ_KV,
            BLOCK_SIZE,
            block_table,
            num_blocks=NUM_BLOCKS,
        )
        v_pa = bnsd_to_v_cache(
            v_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table, num_blocks=NUM_BLOCKS
        )

        cache_info = None
        if NUM_BLOCKS != 0:
            k_pa_for_golden = k_pa.clone()
            v_pa_for_golden = v_pa.clone()
            block_table_for_golden = block_table.copy()
            cache_info = (k_pa_for_golden, v_pa_for_golden, block_table_for_golden)

        # 从 K cache 中提取 deq_k（用于 NPU 调用，共享内存）
        k_pa_f32 = (
            k_pa.view(torch.uint8)
            .view(k_pa.shape[0], k_pa.shape[1], -1)
            .view(torch.float32)
        )
        deq_k_npu = k_pa_f32[:, :, -BLOCK_SIZE:]

        if not IS_CONTIGUOUS:
            kv_cache = torch.stack([k_pa, v_pa], dim=2)
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, :, 0]
            v_pa = kv_cache[:, :, 1]

        output = fa_run_npu(
            q_npu,
            k_pa,
            v_pa,
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
        raise NotImplementedError("当前仅支持 PA 模式")

    atten_out = output[0]
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]

    return output, cache_info


def npu_fp8_full_quant_external(ext):
    """使用外部 NPU 排布数据直接调用 NPU 算子"""
    logger.info("Using external NPU data")
    softmax_scale = 1.0 / math.sqrt(D)
    T = sum(ACTUAL_SEQ_Q)
    out_dtype = OUTPUT_DETYPE

    if max(ACTUAL_SEQ_KV) == 0:
        atten_out = torch.zeros((T, N_q, D), dtype=out_dtype)
        lse_out = torch.full((T, N_q, 1), float("inf"), dtype=torch.float32)
        return (atten_out, lse_out)

    accum_seq_q = (
        make_accum_seq(ACTUAL_SEQ_Q)
        if INPUT_LAYOUT in ("NTD_TND", "TND")
        else ACTUAL_SEQ_Q
    )

    torch_npu.npu.set_device(int(DEVICE_ID))

    q_npu = ext.q_fp8.npu()
    k_npu = ext.k_fp8.npu()
    v_npu = ext.v_fp8.npu()
    deq_q_npu = ext.deq_q.float().npu()
    deq_v_npu = ext.deq_v.float().npu()
    p_scale_npu = ext.p_scale.float().npu()
    block_table_npu = ext.block_table.int().npu()

    k_pa_f32 = (
        k_npu.view(torch.uint8)
        .view(k_npu.shape[0], k_npu.shape[1], -1)
        .view(torch.float32)
    )
    deq_k_npu = k_pa_f32[:, :, -BLOCK_SIZE:]

    if ext.mask is not None:
        mask = ext.mask.bool().npu()
        logger.info("Using external mask, shape: %s", mask.shape)
    elif SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(1, 2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None

    logger.info("[NPU] q dtype: %s, shape: %s", q_npu.dtype, q_npu.shape)
    logger.info("[NPU] k dtype: %s, shape: %s", k_npu.dtype, k_npu.shape)
    logger.info("[NPU] v dtype: %s, shape: %s", v_npu.dtype, v_npu.shape)
    logger.info(
        "[NPU] key is_contiguous: %s, value is_contiguous: %s",
        k_npu.is_contiguous(),
        v_npu.is_contiguous(),
    )

    torch.npu.synchronize()
    atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
        q_npu,
        k_npu[:, :, :128, :],
        v_npu[:, :, :128, :],
        atten_mask=mask,
        actual_seq_qlen=accum_seq_q,
        actual_seq_kvlen=ACTUAL_SEQ_KV,
        dequant_scale_query=deq_q_npu,
        dequant_scale_key=deq_k_npu,
        dequant_scale_value=deq_v_npu,
        block_table=block_table_npu,
        block_size=ext.block_size,
        num_query_heads=N_q,
        num_key_value_heads=N_kv,
        softmax_scale=softmax_scale,
        input_layout=INPUT_LAYOUT,
        sparse_mode=SPARSE_MODE,
        quant_scale_p=p_scale_npu,
        out_dtype=out_dtype,
        query_quant_mode=3,
        key_quant_mode=3,
        value_quant_mode=2,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        return_softmax_lse=ENABLE_LSE,
    )
    torch.npu.synchronize()
    atten_out = atten_out.cpu()
    lse_out = lse_out.cpu()

    T_actual = sum(ACTUAL_SEQ_Q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    return (atten_out, lse_out)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="FIA FullQuant GQA Golden")
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
    parser.add_argument(
        "--use-external-input",
        action="store_true",
        default=False,
        help="使用外部 NPU 排布 pt 文件作为输入",
    )
    parser.add_argument(
        "--load-pt-dir",
        default=None,
        help="外部 pt 文件所在目录（--use-external-input 时生效）",
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
    logger.info("FIA FullQuant GQA Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("=" * 60)
    logger.info(
        "场景: %s, INPUT_LAYOUT=%s, OUTPUT_LAYOUT=%s",
        "PA" if ENABLE_PA else "noPA",
        INPUT_LAYOUT,
        OUTPUT_LAYOUT,
    )
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)

    block_table_torch = None
    if "gen" in mode:
        if args.use_external_input:
            if not args.load_pt_dir:
                parser.error("--use-external-input requires --load-pt-dir")
            external_data = load_external_data.load_data_from_dir(args.load_pt_dir)
            B = external_data.B
            N_q = external_data.N_q
            N_kv = external_data.N_kv
            D = external_data.D
            ACTUAL_SEQ_Q = external_data.ACTUAL_SEQ_Q
            ACTUAL_SEQ_KV = external_data.ACTUAL_SEQ_KV
            BLOCK_SIZE = external_data.block_size
            bnsd_tup = external_to_bnsd(external_data)
            block_table_torch = external_data.block_table
        else:
            bnsd_tup = generate_data()
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
        ) = bnsd_tup
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
                block_table_torch,
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
        cpu_out, cpu_lse = cpu_fp8_fullquant_gqa_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            ACTUAL_SEQ_Q,
            ACTUAL_SEQ_KV,
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
        output, cache_info = npu_fp8_full_quant(
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
        )
        atten_out, lse_out = output
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("[Done] NPU 输出已保存，退出")
        exit(0)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, ACTUAL_SEQ_Q, OUTPUT_LAYOUT)
    result_compare_method.check_result(cpu_tnd_torch, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(cpu_lse, ACTUAL_SEQ_Q, "TND")
        result_compare_method.check_result(cpu_lse_tnd_torch, lse_out)
