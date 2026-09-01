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
QFA FullQuant GQA Golden (FP8全量化)

功能：生成 BNSD 数据 → FP8 per-token-head / per-head 量化 → CPU golden → layout 转换 → NPU 调用 → 精度对比
量化：Q/K per-token-head, V per-head, descale dtype FP32
QFA quant_mode=6 (GQA_FP8_FULLQUANT), layout_q=NTD, layout_kv=PA_BNBD, layout_out=TND, block_size=128, D=128
"""

import argparse
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    _HAS_NPU = True
except ImportError as e:
    _HAS_NPU = False

try:
    from . import result_compare_method
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=float("inf"), linewidth=300, precision=4)

# ==============================================================================
# 配置区
# ==============================================================================
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 2
N_kv = 1
D = 128

ACTUAL_SEQ_Q = [128]
ACTUAL_SEQ_KV = [256]

# QFA layout 属性 (GQA 固定值)
LAYOUT_Q = "NTD"
LAYOUT_Q_DESCALE = "NT"
LAYOUT_KV = "PA_BNBD"
LAYOUT_OUT = "TND"

# PA KV Cache Layout (数据排布，对应 LAYOUT_KV="PA_BNBD")
KV_CACHE_LAYOUT = "BnNBsD"

# Data Range (lo, hi)
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-1.0, 1.0)
V_DATA_RANGE = (-1.0, 1.0)

ENABLE_PA = True
ENABLE_LSE = True
BLOCK_SIZE = 128
MASK_MODE = 3
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

# QFA quant_mode=6 (GQA_FP8_FULLQUANT)
QUANT_MODE = 6

# 物理 block 数量，0 表示使用默认值（等于 total_blocks）
NUM_BLOCKS = 0


# ==============================================================================
# 序列长度转换: actual_seq → QFA cu_seqlens / seqused
# ==============================================================================
def make_cu_seqlens(actual_seq):
    """actual_seq (per-batch length list) → cu_seqlens (prefix sum, prepend 0)"""
    cu = [0]
    for s in actual_seq:
        cu.append(cu[-1] + s)
    return cu


def make_seqused(actual_seq):
    """actual_seq (per-batch length list) → seqused (same as actual_seq)"""
    return list(actual_seq)


# ==============================================================================
# 数据生成函数 (per-token-head / per-head FP8 量化, descale=FP32)
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
    # quant_scale_q = torch.ones_like(quant_scale_q)
    # quant_scale_k = torch.ones_like(quant_scale_k)
    # quant_scale_v = torch.ones_like(quant_scale_v)

    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    dequant_scale_v = (1.0 / quant_scale_v).contiguous()
    # dequant_scale_q = torch.ones_like(dequant_scale_q)
    # dequant_scale_k = torch.ones_like(dequant_scale_k)
    # dequant_scale_v = torch.ones_like(dequant_scale_v)

    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    v_fp8 = quant_fp16_to_fp8(v_fp16, quant_scale_v)
    # q_fp8 = torch.ones_like(q_fp8)
    # k_fp8 = torch.ones_like(k_fp8)
    # v_fp8 = torch.ones_like(v_fp8)

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
# CPU Golden 函数 (与 FIA golden 一致)
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


def printmm(matrix, layout, z_size):
    import torch.nn.functional as F

    d0, d1, d2, d3 = matrix.shape
    if d3 % z_size != 0:
        pad_size = (d3 + z_size - 1) // z_size * z_size - d3
        matrix = F.pad(matrix, (0, pad_size))
        d3 = d3 + pad_size
    if layout == "DN":
        matrix = matrix.permute(0, 1, 3, 2)
    matrix = matrix.reshape(d0, d1, d2, d3 // z_size, z_size)
    matrix = matrix.permute(0, 1, 3, 2, 4)
    # print("---------------matrix print---------------")
    # for val in matrix.flatten():
    #     print(f"{val:.5f}")
    return matrix


def cpu_fp8_fullquant_golden(
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
            # print(f"###{qi=}")
            # print(f"###{kj_T=}")

            # print(f"###{printmm(qi.permute(0,1,3,2), 'ND', 32)=}")
            # print(f"###{printmm(kj_T, 'ND', 32)=}")
            # print(f"###{sij.permute(0,1,3,2)=}")
            deq_qi = deq_qi * softmax_scale
            sij = sij * deq_qi * deq_kj_T
            # print(f"###{deq_qi=}")
            # print(f"###{deq_kj_T=}")
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
            # print(f"###{pij_drop=}")
            # print(f"###{printmm(pij_drop, 'ND', 32)=}")
            # print(f"###{vj=}")
            # print(f"###{printmm(vj.permute(0,1,3,2), 'ND', 32)=}")
            # print(f"###{pij_v=}")
            # print(f"###{printmm(pij_v, 'ND', 32)=}")

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
    """BNSD → QFA layout (NTD/TND/BNSD/BSND)"""
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
    elif layout == "NTD":
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
    """Scale to QFA layout"""
    tensor = tensor.cpu().contiguous()
    if scale_type == "deq_q":
        b, n, _, _ = tensor.shape
        T = sum(seq_lens)
        if LAYOUT_Q_DESCALE == "NT":
            result = torch.zeros((n, T), dtype=torch.float32)
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                for n_idx in range(n):
                    result[n_idx, t : t + act_s] = tensor[b_idx, n_idx, :act_s, 0]
                t += act_s
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
# NPU 调用 - QFA 双算子接口
# ==============================================================================
class Network(nn.Module):
    """aclgraph 编译目标: forward 只包含两个 torch.library op 调用。"""

    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        mask,
        cu_seqlens_q_t,
        seqused_q_t,
        seqused_kv_t,
        dequant_scale_q,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        metadata = quant_flash_attn_metadata(
            num_heads_q=q_n,
            num_heads_kv=kv_n,
            head_dim=q.shape[-1],
            quant_mode=QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_kv=None,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            head_dim_v=None,
            mask_mode=MASK_MODE,
            layout_q=LAYOUT_Q,
            layout_q_descale=LAYOUT_Q_DESCALE,
            layout_kv=LAYOUT_KV,
            layout_out=LAYOUT_OUT,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )

        atten_out, lse_out = quant_flash_attn(
            q=q,
            k=k,
            v=v,
            q_descale=dequant_scale_q,
            k_descale=dequant_scale_k,
            v_descale=dequant_scale_v,
            quant_mode=QUANT_MODE,
            block_table=block_table,
            p_scale=p_scale,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_kv=None,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            mask_mode=MASK_MODE,
            layout_q=LAYOUT_Q,
            layout_q_descale=LAYOUT_Q_DESCALE,
            layout_kv=LAYOUT_KV,
            layout_out=LAYOUT_OUT,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            return_softmax_lse=ENABLE_LSE,
        )
        return atten_out, lse_out


def _call_npu_qfa_op(
    q,
    k,
    v,
    mask,
    cu_seqlens_q,
    seqused_q,
    seqused_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    max_seqlen_q,
    max_seqlen_kv,
):
    """调用 NPU QFA 双算子 (quant_flash_attn_metadata + quant_flash_attn)"""
    if not _HAS_NPU:
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )

    cu_seqlens_q_t = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
        if cu_seqlens_q is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=torch.int32).npu()
        if seqused_q is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=torch.int32).npu()
        if seqused_kv is not None
        else None
    )

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=q_n,
        num_heads_kv=kv_n,
        head_dim=q.shape[-1],
        quant_mode=QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        mask_mode=MASK_MODE,
        layout_q=LAYOUT_Q,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=LAYOUT_KV,
        layout_out=LAYOUT_OUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    # if MASK_MODE == 3:
    #     metadata = torch.load("./metadata_sp3.pt", weights_only=False)
    # elif MASK_MODE == 0:
    #     metadata = torch.load("./metadata_sp0.pt", weights_only=False)

    atten_out, lse_out = quant_flash_attn(
        q=q,
        k=k,
        v=v,
        q_descale=dequant_scale_q,
        k_descale=dequant_scale_k,
        v_descale=dequant_scale_v,
        quant_mode=QUANT_MODE,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=MASK_MODE,
        layout_q=LAYOUT_Q,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=LAYOUT_KV,
        layout_out=LAYOUT_OUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        return_softmax_lse=ENABLE_LSE,
    )
    torch.npu.synchronize()
    return atten_out, lse_out


def qfa_fp8_torch_npu(
    q,
    k,
    v,
    mask,
    cu_seqlens_q,
    seqused_q,
    seqused_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    max_seqlen_q,
    max_seqlen_kv,
):
    """NPU 调用入口, 支持 GRAPH_PATH=0 (单算子) 和 GRAPH_PATH=7 (aclgraph)"""
    if GRAPH_PATH == 0:
        logger.info("[NPU] GRAPH_PATH == 0, 单算子模式...")
        return _call_npu_qfa_op(
            q,
            k,
            v,
            mask,
            cu_seqlens_q,
            seqused_q,
            seqused_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            max_seqlen_q,
            max_seqlen_kv,
        )

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    with torch.no_grad():
        torch.npu.synchronize()

        cu_seqlens_q_t = (
            torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
            if cu_seqlens_q is not None
            else None
        )
        seqused_q_t = (
            torch.tensor(seqused_q, dtype=torch.int32).npu()
            if seqused_q is not None
            else None
        )
        seqused_kv_t = (
            torch.tensor(seqused_kv, dtype=torch.int32).npu()
            if seqused_kv is not None
            else None
        )

        fa_args = (
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            max_seqlen_q,
            max_seqlen_kv,
        )

        logger.info("[NPU] 调用 aclgraph (npugraph_ex)...")
        npu_backend = "npugraph_ex"
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend=npu_backend, dynamic=False
        )
        for t in (
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_v,
            p_scale,
            block_table,
        ):
            if t is not None:
                torch._dynamo.mark_static(t)
        atten_out, lse_out = npu_mode(*fa_args)

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out


def _build_mask():
    if MASK_MODE == 0:
        return None
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def prepare_npu_inputs_gqa_fp8(
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
    """准备 NPU 侧入参 (GQA FP8, 仅 PA).

    返回 dict 的 key 与 fa_run_npu / _call_npu_qfa_op 形参名一一对应:
      q, k, v, mask, cu_seqlens_q, seqused_q, seqused_kv,
      dequant_scale_q, dequant_scale_k, dequant_scale_v,
      p_scale, block_table, block_size, q_n, kv_n,
      softmax_scale, max_seqlen_q, max_seqlen_kv
    """
    softmax_scale = 1.0 / math.sqrt(D)
    cu_seqlens_q = make_cu_seqlens(actual_seq_q)
    seqused_q = make_seqused(actual_seq_q)
    seqused_kv = make_seqused(actual_seq_kv)
    max_seqlen_q = max(actual_seq_q) if actual_seq_q else -1
    max_seqlen_kv = max(actual_seq_kv) if actual_seq_kv else -1

    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, LAYOUT_Q)
    deq_q_npu = convert_scale_to_layout(dequant_scale_q, actual_seq_q, "deq_q")
    deq_v_npu = convert_scale_to_layout(dequant_scale_v, actual_seq_kv, "deq_v")
    mask_arg = _build_mask()

    if block_table_torch is not None:
        block_table = block_table_torch.cpu().numpy().astype(np.int32).copy()
        block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)
    else:
        block_table = create_block_table(
            actual_seq_kv, BLOCK_SIZE, num_blocks=NUM_BLOCKS
        )
        block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)

    k_pa = bnsd_to_k_cache(
        k_fp8,
        dequant_scale_k,
        actual_seq_kv,
        BLOCK_SIZE,
        block_table,
        num_blocks=NUM_BLOCKS,
    )
    v_pa = bnsd_to_v_cache(
        v_fp8, actual_seq_kv, BLOCK_SIZE, block_table, num_blocks=NUM_BLOCKS
    )

    if not IS_CONTIGUOUS:
        kv_cache = torch.stack([k_pa, v_pa], dim=2)
        kv_cache = kv_cache.npu()
        k_pa = kv_cache[:, :, 0]
        v_pa = kv_cache[:, :, 1]

    return dict(
        q=q_npu,
        k=k_pa,
        v=v_pa,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=dequant_scale_k,
        dequant_scale_v=deq_v_npu,
        p_scale=p_scale,
        block_table=block_table_tensor,
        block_size=BLOCK_SIZE,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )


def fa_run_npu(
    q,
    k,
    v,
    mask,
    cu_seqlens_q,
    seqused_q,
    seqused_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    block_size,
    q_n,
    kv_n,
    softmax_scale,
    max_seqlen_q,
    max_seqlen_kv,
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
        mask = mask.npu()

    # 从 cache 中取数据切片 (K cache 含 scale rows, 取前 block_size 行为 K 数据)
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
    # k = k.contiguous().npu()
    # v = v.contiguous().npu()
    # dequant_scale_k = dequant_scale_k.contiguous().npu()
    logger.info(
        "[NPU] layout_q: %s, layout_kv: %s, mask_mode: %s",
        LAYOUT_Q,
        LAYOUT_KV,
        MASK_MODE,
    )
    logger.info("[NPU] k is_contiguous: %s, stride: %s", k.is_contiguous(), k.stride())
    logger.info("[NPU] v is_contiguous: %s, stride: %s", v.is_contiguous(), v.stride())
    logger.info(
        "[NPU] dequant_scale_k is_contiguous: %s, stride: %s",
        dequant_scale_k.is_contiguous(),
        dequant_scale_k.stride(),
    )

    atten_out, lse_out = qfa_fp8_torch_npu(
        q,
        k,
        v,
        mask,
        cu_seqlens_q,
        seqused_q,
        seqused_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        max_seqlen_q,
        max_seqlen_kv,
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
    """主 NPU 量化函数 - 准备数据并调用 NPU QFA 双算子"""
    if not ENABLE_PA:
        raise NotImplementedError("QFA GQA 仅支持 PA 模式")

    inputs = prepare_npu_inputs_gqa_fp8(
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        actual_seq_q,
        actual_seq_kv,
        block_table_torch=block_table_torch,
    )

    # NUM_BLOCKS != 0 时需保留 cache 副本供 golden 重建
    cache_info = None
    if NUM_BLOCKS != 0:
        k_pa_clone = inputs["k"].clone()
        v_pa_clone = inputs["v"].clone()
        bt_clone = inputs["block_table"].cpu().numpy().copy()
        cache_info = (k_pa_clone, v_pa_clone, bt_clone)

    output = fa_run_npu(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["mask"],
        inputs["cu_seqlens_q"],
        inputs["seqused_q"],
        inputs["seqused_kv"],
        inputs["dequant_scale_q"],
        inputs["dequant_scale_k"],
        inputs["dequant_scale_v"],
        inputs["p_scale"],
        inputs["block_table"],
        inputs["block_size"],
        inputs["q_n"],
        inputs["kv_n"],
        inputs["softmax_scale"],
        inputs["max_seqlen_q"],
        inputs["max_seqlen_kv"],
    )

    atten_out = output[0]
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]

    return output, cache_info


# ==============================================================================
# PA cache → BNSD 还原 (NUM_BLOCKS != 0 时用)
# ==============================================================================
def _bnbd_to_bnsd(kv_bnbd, block_table, actual_seq_kv, block_size):
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


def _bnb_to_bns1(k_scale_bnb, block_table, actual_seq_kv, block_size):
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
    """从 PA cache 还原 BNSD 格式的 K/V/deq_k"""
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


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="QFA FullQuant GQA Golden")
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
    logger.info("QFA FullQuant GQA Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("=" * 60)
    logger.info(
        "场景: %s, LAYOUT_Q=%s, LAYOUT_KV=%s, LAYOUT_OUT=%s",
        "PA" if ENABLE_PA else "noPA",
        LAYOUT_Q,
        LAYOUT_KV,
        LAYOUT_OUT,
    )
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)

    block_table_torch = None
    if "gen" in mode:
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
        cpu_out, cpu_lse = cpu_fp8_fullquant_golden(
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
    cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, ACTUAL_SEQ_Q, LAYOUT_OUT)
    result_compare_method.check_result(cpu_tnd_torch, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(cpu_lse, ACTUAL_SEQ_Q, "TND")
        cpu_lse_nt_torch = cpu_lse_tnd_torch.squeeze(-1).permute(1, 0).contiguous()
        result_compare_method.check_result(cpu_lse_nt_torch, lse_out)
