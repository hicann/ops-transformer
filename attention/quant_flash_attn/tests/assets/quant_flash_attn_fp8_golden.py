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
GQA FP8 全量化 Golden (quant_mode=6)

功能: 生成 BNSD 数据 → FP8 per-token-head / per-head 量化 → CPU golden → layout 转换 → NPU 调用
量化: Q/K per-token-head, V per-head, descale dtype FP32 (非 e8m0)
layout_q=NTD, layout_q_descale=NT, layout_kv=PA_BNBD (K cache 含末 4 行 FP32 scale),
layout_out=TND, 仅 PA 模式, GQA (N_q != N_kv 时内部 broadcast)

TTK 适配: 模块级全局变量 (B/N_q/N_kv/D/ENABLE_PA/...) 默认 None,
由 wrapper._apply_golden_globals 从 csv attributes 注入 (与 quant_flash_attn_golden.py 同机制)。
本文件仅含 GQA FP8 全量化路径, MXFP8 路径见 quant_flash_attn_golden.py。
"""

import logging
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    _HAS_NPU = True
except ImportError as e:
    _HAS_NPU = False

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=float("inf"), linewidth=300, precision=4)


# ==============================================================================
# 配置区 — 所有值由 wrapper._apply_golden_globals 从 csv attributes 注入
# 默认 None; csv 未提供的 key → wrapper 用 dict.get() 得 None → 透传 None 给 aclnn
# 零内部推导: 不在此处硬编码任何 op 参数
# ==============================================================================
GRAPH_PATH = int(os.environ.get("GRAPH_PATH", "0"))

B = None
BATCH_SIZE = None
N_q = None
N_kv = None
D = None

ACTUAL_SEQ_Q = None
ACTUAL_SEQ_KV = None
CU_SEQLENS_Q = None
CU_SEQLENS_KV = None
SEQUSED_Q = None
SEQUSED_KV = None
MAX_SEQLEN_Q = None
MAX_SEQLEN_KV = None

ENABLE_PA = None
BLOCK_SIZE = None

SPARSE_MODE = 0
FP8_DTYPE = torch.float8_e4m3fn

# QFA layout 属性 (GQA FP8 固定值, 但保留可注入以兼容 csv 覆盖)
LAYOUT_Q = "NTD"
LAYOUT_Q_DESCALE = "NT"
LAYOUT_KV = "PA_BNBD"
LAYOUT_OUT = "TND"

# PA KV Cache Layout (数据排布, 对应 LAYOUT_KV="PA_BNBD")
KV_CACHE_LAYOUT = "BnNBsD"

Q_SCALE_LAYOUT = None
P_SCALE = None

SOFTMAX_SCALE = None
IS_CONTIGUOUS = None

ENABLE_LSE = None

SEED_Q = None
SEED_K = None
SEED_V = None
SEED_BLOCK_TABLE = 1234

DATA_RANGE_Q = None
DATA_RANGE_K = None
DATA_RANGE_V = None

DEVICE_ID = None

OUTPUT_DETYPE = torch.bfloat16

# Q/K cache 末尾附加 scale 行数 (FP32 K scale 以 uint8 view 存入)
K_SCALE_ROWS = 4

EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

# QFA quant_mode=6 (GQA_FP8_FULLQUANT)
QUANT_MODE = 6

# 物理 block 数量, 0 表示使用默认值 (等于 total_blocks)
NUM_BLOCKS = 0


# ==============================================================================
# 序列长度转换
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


def _actual_seq_from_cu_seqlens(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _actual_seq_q():
    """actual_seq_q (csv SEQUSED_Q 优先, 否则从 CU_SEQLENS_Q 差分还原)."""
    return (
        SEQUSED_Q
        if SEQUSED_Q is not None
        else _actual_seq_from_cu_seqlens(CU_SEQLENS_Q)
    )


def _actual_seq_kv():
    """actual_seq_kv (csv SEQUSED_KV 优先, 否则从 CU_SEQLENS_KV 差分还原)."""
    return (
        SEQUSED_KV
        if SEQUSED_KV is not None
        else _actual_seq_from_cu_seqlens(CU_SEQLENS_KV)
    )


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    """GQA broadcast: [B, N_kv, S, D] → [B, N_q, S, D] (N_q = N_kv * factor)."""
    if num_heads == num_kv_heads:
        return kv_tensor.contiguous()
    factor = num_heads // num_kv_heads
    return kv_tensor.repeat_interleave(factor, dim=1).contiguous()


# ==============================================================================
# FP8 量化 (per-token-head Q/K, per-head V, descale=FP32)
# ==============================================================================
def get_fp8_per_token_head_quant_scale(tensor):
    """per-token-head quant scale: shape (B, N, S, 1) FP32."""
    tensor = tensor.contiguous()
    B, N, S, _ = tensor.shape
    fp8_e4m3_max = 448.0
    row_max = torch.abs(tensor).max(dim=3, keepdim=True).values
    row_max = torch.max(row_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / row_max
    return scale.view(B, N, S, 1).float().contiguous()


def get_fp8_per_head_quant_scale(tensor):
    """per-head quant scale: shape (1, N, 1, 1) FP32."""
    tensor = tensor.contiguous()
    fp8_e4m3_max = 448.0
    head_max = torch.abs(tensor).amax(dim=(0, 2, 3), keepdim=True)
    head_max = torch.max(head_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / head_max
    return scale.float().contiguous()


def quant_fp16_to_fp8(tensor, scale):
    """将 fp16/bf16 数据量化为 fp8_e4m3 (tensor * scale → clamp → fp8)."""
    tensor = tensor.contiguous()
    scale = scale.contiguous()
    result = tensor.float() * scale
    result = torch.clamp(result, -448.0, 448.0)
    return result.to(FP8_DTYPE).contiguous()


def fp8_quant_scales_to_descales(quant_scale_q, quant_scale_k, quant_scale_v):
    """quant_scale → dequant_scale (1/scale), FP32. 供 wrapper 分流调用."""
    deq_q = (1.0 / quant_scale_q).contiguous().float()
    deq_k = (1.0 / quant_scale_k).contiguous().float()
    deq_v = (1.0 / quant_scale_v).contiguous().float()
    return deq_q, deq_k, deq_v


# ==============================================================================
# PA K/V cache (BNSD → PA_BNBD, K cache 末 K_SCALE_ROWS 行存 FP32 deq_k)
# ==============================================================================
def bnsd_to_k_cache_gqa(
    k_fp8_bnsd, k_scale_fp32_bnsd, seq_lens, block_size, block_table, num_blocks=0
):
    """BNSD → PA K cache [Bn,N,block_size+K_SCALE_ROWS,D] FP8, 末 4 行存 FP32 deq_k.

    K 数据 FP8, K scale FP32 (per-token-head, shape [B,N,S,1]) 嵌入末 4 行:
      scale_buf: [N, K_SCALE_ROWS, D//4] FP32 → uint8 view → [N, K_SCALE_ROWS, D]
      末 block_size 个 FP32 值为有效 scale (与 valid token 数对齐)
    """
    k_fp8_bnsd = k_fp8_bnsd.contiguous()
    k_scale_fp32_bnsd = k_scale_fp32_bnsd.contiguous()
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape
    scale_rows = K_SCALE_ROWS
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


def bnsd_to_v_cache_gqa(tensor_bnsd, seq_lens, block_size, block_table, num_blocks=0):
    """BNSD → PA V cache [Bn,N,block_size+K_SCALE_ROWS,D] FP8 (末 4 行占位无 scale)."""
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)
    cache_blocks = num_blocks if num_blocks != 0 else total_blocks

    out_cache = torch.zeros(
        (cache_blocks, heads, block_size + K_SCALE_ROWS, dim),
        dtype=FP8_DTYPE,
        device=device,
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


# ==============================================================================
# PA cache → BNSD 还原 (golden 用, 从 PA cache 还原 K/V/deq_k 供 cpu golden)
# ==============================================================================
def _bnbd_to_bnsd_gqa(kv_bnbd, block_table, actual_seq_kv, block_size):
    """PA [Bn,N,Bs,D] → BNSD [B,N,max_skv,D] (按 block_table scatter)."""
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


def _bnb_to_bns1_gqa(k_scale_bnb, block_table, actual_seq_kv, block_size):
    """PA K scale [Bn,N,Bs] FP32 → BNSD [B,N,max_skv,1] FP32 (从 K cache 末行提取)."""
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


def pa_cache_to_bnsd_gqa(k_pa, v_pa, block_table, actual_seq_kv, block_size):
    """从 PA cache 还原 BNSD 格式的 K/V/deq_k.

    入参:
      k_pa: [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8 (末 K_SCALE_ROWS 行存 FP32 deq_k)
      v_pa: [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
    返回: (k_bnsd, v_bnsd, deq_k_bns1)
      k_bnsd: [B,N_kv,max_skv,D] FP8
      v_bnsd: [B,N_kv,max_skv,D] FP8
      deq_k_bns1: [B,N_kv,max_skv,1] FP32
    """
    k_data = k_pa[:, :, :block_size, :].contiguous()
    v_data = v_pa[:, :, :block_size, :].contiguous()
    # 从 K cache 末 block_size 个 FP32 值提取 deq_k
    # k_pa: [Bn,N,block_size+K_SCALE_ROWS,D] FP8 → uint8 → reshape FP32
    k_pa_f32 = (
        k_pa.view(torch.uint8)
        .view(k_pa.shape[0], k_pa.shape[1], -1)
        .view(torch.float32)
    )
    deq_k_flat = k_pa_f32[:, :, -block_size:].contiguous()
    k_bnsd = _bnbd_to_bnsd_gqa(k_data, block_table, actual_seq_kv, block_size)
    v_bnsd = _bnbd_to_bnsd_gqa(v_data, block_table, actual_seq_kv, block_size)
    deq_k_bns1 = _bnb_to_bns1_gqa(deq_k_flat, block_table, actual_seq_kv, block_size)
    return k_bnsd, v_bnsd, deq_k_bns1


# ==============================================================================
# Layout 转换 (BNSD → NTD / TND; scale → NT / [N_kv])
# ==============================================================================
def convert_q_bnsd_to_ntd(tensor_bnsd, seq_lens):
    """BNSD [B,N,S,D] → NTD [N,T,D] (T = sum(seq_lens))."""
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    tensor = tensor.cpu().contiguous()
    b, n, _, d = tensor.shape
    T = sum(seq_lens)
    result = torch.zeros((n, T, d), dtype=tensor.dtype, device=tensor.device)
    t = 0
    for b_idx in range(b):
        act_s = seq_lens[b_idx]
        for n_idx in range(n):
            result[n_idx, t : t + act_s, :] = tensor[b_idx, n_idx, :act_s, :]
        t += act_s
    return result.contiguous()


def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    """BNSD → QFA layout (NTD/TND/BNSD/BSND) — golden 输出对齐 NPU layout_out 用."""
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    tensor = tensor.cpu().contiguous()
    b, n, _, d = tensor.shape
    max_org_s = max(seq_lens) if seq_lens else 0

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, n, d), dtype=tensor.dtype, device=tensor.device)
        if cu_seqlens is not None:
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                if act_s <= 0:
                    continue
                offset = cu_seqlens[b_idx]
                result[offset : offset + act_s, :, :] = tensor[
                    b_idx, :, :act_s, :
                ].permute(1, 0, 2)
        else:
            t = 0
            for b_idx in range(b):
                act_s = seq_lens[b_idx]
                if act_s > 0:
                    result[t : t + act_s, :, :] = tensor[b_idx, :, :act_s, :].permute(
                        1, 0, 2
                    )
                t += act_s
        return result.contiguous()
    elif layout == "NTD":
        return convert_q_bnsd_to_ntd(tensor, seq_lens)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_scale_to_layout_gqa(tensor, seq_lens, scale_type):
    """Scale BNSD → QFA GQA layout.

    scale_type="deq_q": [B,N,S,1] FP32 → NT [N,T] (T=sum(seq_lens))
    scale_type="deq_v": [1,N_kv,1,1] FP32 → [N_kv] FP32
    scale_type="deq_k": [B,N_kv,S,1] FP32 → 透传 (BNSD, NPU 从 K cache 提取)
    """
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


def ntd_to_bnsd_q_gqa(tensor_ntd, seq_lens):
    """NTD [N,T,D] → BNSD [B,N,max_sq,D] (T 沿 b 累加)."""
    tensor = (
        tensor_ntd
        if isinstance(tensor_ntd, torch.Tensor)
        else torch.as_tensor(tensor_ntd)
    )
    tensor = tensor.cpu().contiguous()
    n, T, d = tensor.shape
    b = len(seq_lens)
    max_sq = max(seq_lens) if seq_lens else 0
    result = torch.zeros((b, n, max_sq, d), dtype=tensor.dtype, device=tensor.device)
    t = 0
    for b_idx in range(b):
        act_s = seq_lens[b_idx]
        if act_s > 0:
            for n_idx in range(n):
                result[b_idx, n_idx, :act_s, :] = tensor[n_idx, t : t + act_s, :]
        t += act_s
    return result.contiguous()


def nt_to_bnsd_q_scale_gqa(tensor_nt, seq_lens):
    """NT [N,T] FP32 → BNSD [B,N,max_sq,1] FP32."""
    tensor = (
        tensor_nt if isinstance(tensor_nt, torch.Tensor) else torch.as_tensor(tensor_nt)
    )
    tensor = tensor.cpu().contiguous().float()
    n, T = tensor.shape
    b = len(seq_lens)
    max_sq = max(seq_lens) if seq_lens else 0
    result = torch.zeros((b, n, max_sq, 1), dtype=torch.float32, device=tensor.device)
    t = 0
    for b_idx in range(b):
        act_s = seq_lens[b_idx]
        if act_s > 0:
            for n_idx in range(n):
                result[b_idx, n_idx, :act_s, 0] = tensor[n_idx, t : t + act_s]
        t += act_s
    return result.contiguous()


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    """TND [T,N,D] padding 位置填 fill_value (匹配 NPU 行为)."""
    tensor = tensor_tnd.contiguous()
    T, N, D = tensor.shape
    result = tensor.clone()
    if cu_seqlens is not None:
        for b_idx in range(len(seq_lens)):
            act_s = seq_lens[b_idx]
            offset = cu_seqlens[b_idx]
            pad_start = offset + act_s
            pad_end = cu_seqlens[b_idx + 1] if b_idx + 1 < len(cu_seqlens) else T
            if pad_end > pad_start:
                result[pad_start:pad_end, :, :] = fill_value
    else:
        t = 0
        for b_idx in range(len(seq_lens)):
            act_s = seq_lens[b_idx]
            t += act_s
        if T > t:
            result[t:, :, :] = fill_value
    return result.contiguous()


# ==============================================================================
# CPU Golden (per-block flash attention, descale 不做 group 维扩展)
# ==============================================================================
def get_softmax_scale(scale_value, head_dim):
    if scale_value is not None:
        return float(scale_value)
    return 1.0 / math.sqrt(head_dim)


def cpu_fp8_fullquant_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    deq_q,
    deq_k,
    deq_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
):
    """CPU golden reference — 所有操作在 CPU 上执行.

    入参均为 BNSD 布局:
      q_fp8: [B,N_q,max_sq,D] FP8
      k_fp8: [B,N_q,max_skv,D] FP8 (GQA 已 broadcast)
      v_fp8: [B,N_q,max_skv,D] FP8
      deq_q: [B,N_q,max_sq,1] FP32
      deq_k: [B,N_q,max_skv,1] FP32
      deq_v: [1,N_q,1,1] FP32 (或 broadcast 后 [B,N_q,max_skv,1])
      p_scale: [1] FP32
    返回 (result_bnsd, lse_bnsd)
    """
    ss = get_softmax_scale(
        softmax_scale if softmax_scale is not None else SOFTMAX_SCALE, D
    )
    q_tensor = q_fp8.cpu().to(torch.float32).contiguous()
    batch, heads, q_seq, d_dim = q_tensor.shape

    k_tensor = k_fp8.cpu().to(torch.float32).contiguous()
    v_tensor = v_fp8.cpu().to(torch.float32).contiguous()
    deq_q = deq_q.cpu().float().contiguous()
    deq_k = deq_k.cpu().float().contiguous()
    deq_v = deq_v.cpu().float().contiguous()

    # GQA broadcast (若 N_q != N_kv, 调用方应已 broadcast; 这里再保险一次)
    if N_q != N_kv and k_tensor.shape[1] == N_kv:
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        deq_k = broadcast_kv(N_q, N_kv, deq_k)
        deq_v = broadcast_kv(N_q, N_kv, deq_v)

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
    # 0xFF7FFFFF = FP32 最小有限值 = -3.402823466e38
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
            deq_qi = deq_qi * ss
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
# NPU 调用 — QFA 双算子接口 (metadata + main op)
# ==============================================================================
def _build_mask():
    if SPARSE_MODE == 0:
        return None
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


class Network(nn.Module):
    """aclgraph 编译目标: forward 只包含两个 torch.library op 调用."""

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
        metadata = quant_flash_attn_metadata(
            num_heads_q=q_n,
            num_heads_kv=kv_n,
            head_dim=q.shape[-1],
            quant_mode=QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_kv=None,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            v_descale=dequant_scale_v,
            batch_size=BATCH_SIZE,
            mask_mode=SPARSE_MODE,
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
            mask_mode=SPARSE_MODE,
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
    cu_seqlens_q_t,
    seqused_q_t,
    seqused_kv_t,
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
    """调用 NPU QFA 双算子 (quant_flash_attn_metadata + quant_flash_attn) — 单算子模式."""
    if not _HAS_NPU:
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )

    # cu_seqlens/seqused 直接使用 CSV tensor slot (_t) 的 NPU tensor（保留 dtype）。
    # 兼容旧调用：若 _t 为 None 则从 list 重建 int32 tensor。
    def _as_tensor(t, lst):
        if t is not None:
            return t
        if lst is None:
            return None
        return torch.tensor(list(lst), dtype=torch.int32).npu()

    cu_seqlens_q_t = _as_tensor(cu_seqlens_q_t, None)
    seqused_q_t = _as_tensor(seqused_q_t, None)
    seqused_kv_t = _as_tensor(seqused_kv_t, None)

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=q_n,
        num_heads_kv=kv_n,
        head_dim=D,
        quant_mode=QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        v_descale=dequant_scale_v,
        mask_mode=SPARSE_MODE,
        batch_size=BATCH_SIZE,
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
        mask_mode=SPARSE_MODE,
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
    cu_seqlens_q_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """NPU 调用入口, 支持 GRAPH_PATH=0 (单算子) 和 GRAPH_PATH=7 (aclgraph)."""
    if GRAPH_PATH == 0:
        logger.info("[NPU GQA FP8] GRAPH_PATH == 0, 单算子模式...")
        return _call_npu_qfa_op(
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t if cu_seqlens_q_t is not None else cu_seqlens_q,
            seqused_q_t if seqused_q_t is not None else seqused_q,
            seqused_kv_t if seqused_kv_t is not None else seqused_kv,
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

        # cu_seqlens/seqused 直接使用 CSV tensor slot (_t) 的 NPU tensor（保留 dtype）；
        # 兼容旧调用：_t 为 None 时从 list 重建 int32 tensor。
        def _build_t(t, lst):
            if t is not None:
                return t
            if lst is None:
                return None
            return torch.tensor(list(lst), dtype=torch.int32).npu()

        cu_seqlens_q_t = _build_t(cu_seqlens_q_t, cu_seqlens_q)
        seqused_q_t = _build_t(seqused_q_t, seqused_q)
        seqused_kv_t = _build_t(seqused_kv_t, seqused_kv)

        fa_args = (
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            seqused_q_t,
            seqused_kv_t,
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

        logger.info("[NPU GQA FP8] 调用 aclgraph (npugraph_ex)...")
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
            dequant_scale_k,
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
    cu_seqlens_q_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """将数据转移到 NPU 上并调用 NPU 算子.

    K cache 含 scale rows, deq_k 从 K cache 末 BLOCK_SIZE 行提取 (uint8 → FP32 view).
    """
    torch_npu.npu.set_device(int(DEVICE_ID))

    q = q.npu()
    k = k.npu()
    v = v.npu()

    # 从 K cache 的 scale rows 中提取 deq_k (共享内存)
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
    k = k[:, :, :BLOCK_SIZE, :]
    v = v[:, :, :BLOCK_SIZE, :]

    logger.info("[NPU GQA FP8] q dtype: %s, shape: %s", q.dtype, q.shape)
    logger.info("[NPU GQA FP8] k dtype: %s, shape: %s", k.dtype, k.shape)
    logger.info("[NPU GQA FP8] v dtype: %s, shape: %s", v.dtype, v.shape)
    logger.info(
        "[NPU GQA FP8] deq_q: %s, deq_k: %s, deq_v: %s",
        dequant_scale_q.shape,
        dequant_scale_k.shape,
        dequant_scale_v.shape,
    )
    logger.info(
        "[NPU GQA FP8] layout_q=%s, layout_kv=%s, mask_mode=%s",
        LAYOUT_Q,
        LAYOUT_KV,
        SPARSE_MODE,
    )
    logger.info(
        "[NPU GQA FP8] k is_contiguous: %s, stride: %s",
        k.is_contiguous(),
        k.stride(),
    )
    logger.info(
        "[NPU GQA FP8] v is_contiguous: %s, stride: %s",
        v.is_contiguous(),
        v.stride(),
    )
    logger.info(
        "[NPU GQA FP8] dequant_scale_k is_contiguous: %s, stride: %s",
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
        cu_seqlens_q_t=cu_seqlens_q_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
    )

    if GRAPH_PATH == 0:
        atten_out = atten_out.cpu()
        lse_out = lse_out.cpu()
    return atten_out, lse_out


def prepare_npu_inputs_gqa_fp8(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
):
    """准备 NPU 侧入参 (GQA FP8, 仅 PA).

    返回字典的 key 与 fa_run_npu / _call_npu_qfa_op 形参名一一对应:
      q, k, v, mask, cu_seqlens_q, seqused_kv, max_seqlen_q, max_seqlen_kv,
      dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table,
      q_n, kv_n, softmax_scale, block_size

    入参 q/deq_q 为 NTD/NT layout (final), K/V 为 PA cache (含 scale rows).
    """
    torch_npu.npu.set_device(int(DEVICE_ID))
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    softmax_scale = SOFTMAX_SCALE if SOFTMAX_SCALE is not None else (1.0 / math.sqrt(D))

    q_npu = q_fp8.contiguous().view(FP8_DTYPE).npu()
    deq_q_npu = dequant_scale_q.npu()
    p_scale_npu = p_scale.npu()
    mask_arg = _build_mask()

    if not ENABLE_PA:
        raise NotImplementedError("GQA FP8 (quant_mode=6) 仅支持 PA 模式")

    k_npu = k_fp8.contiguous().view(FP8_DTYPE).npu()
    v_npu = v_fp8.contiguous().view(FP8_DTYPE).npu()
    deq_v_npu = dequant_scale_v.npu()

    if not IS_CONTIGUOUS:
        kv_cache = torch.stack([k_fp8, v_fp8], dim=2)
        kv_cache = kv_cache.npu()
        k_npu = kv_cache[:, :, 0]
        v_npu = kv_cache[:, :, 1]

    block_table_npu = (
        block_table_torch.npu()
        if isinstance(block_table_torch, torch.Tensor)
        else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
    )

    logger.info("[NPU GQA FP8 PA] kv_layout=%s", KV_CACHE_LAYOUT)
    logger.info(
        "[NPU GQA FP8 PA] k=%s, v=%s, deq_q=%s, deq_v=%s",
        k_npu.shape,
        v_npu.shape,
        deq_q_npu.shape,
        deq_v_npu.shape,
    )

    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=dequant_scale_k,
        dequant_scale_v=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=block_table_npu,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q=LAYOUT_Q,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=LAYOUT_KV,
        layout_out=LAYOUT_OUT,
        block_size=BLOCK_SIZE,
        sparse_mode=SPARSE_MODE,
        out_dtype=OUTPUT_DETYPE,
    )


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
    """主 NPU 量化函数 — 准备数据并调用 NPU QFA 双算子 (GQA FP8)."""
    softmax_scale = SOFTMAX_SCALE if SOFTMAX_SCALE is not None else (1.0 / math.sqrt(D))

    cu_seqlens_q = make_cu_seqlens(actual_seq_q)
    seqused_kv = make_seqused(actual_seq_kv)
    max_seqlen_q = max(actual_seq_q) if actual_seq_q else -1
    max_seqlen_kv = max(actual_seq_kv) if actual_seq_kv else -1

    # 入参已是 final layout (q=NTD, deq_q=NT, deq_v=[N_kv], K/V=PA cache)
    mask_arg = _build_mask()

    block_table = block_table_torch
    if block_table is None:
        # 兜底: 从 CSV slot 由 wrapper 传入, 这里不应到达
        raise ValueError("[NPU GQA FP8] block_table_torch is None (PA 模式必需)")

    output = fa_run_npu(
        q_fp8,
        k_fp8,
        v_fp8,
        mask_arg,
        cu_seqlens_q,
        seqused_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        BLOCK_SIZE,
        N_q,
        N_kv,
        softmax_scale,
        max_seqlen_q,
        max_seqlen_kv,
    )

    return output


def npu_gqa_fp8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    cu_seqlens_q_t=None,
    cu_seqlens_kv_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """GQA FP8 NPU 入口 (qfa_wrapper 调用).

    入参为 final-layout fp8 + FP32 descale (inputs.py 已 in-place 写入 slot):
      q: NTD [N,T,D] FP8
      k: PA K cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
      v: PA V cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
      dequant_scale_q: NT [N,T] FP32
      dequant_scale_k: PA 块状 FP32 (与 K cache 数据部分对齐, NPU 从 K cache 提取)
      dequant_scale_v: [N_kv] FP32
      p_scale: [1] FP32
      block_table: [B,max_blocks] int32
    返回 (atten_out, lse_out): atten_out TND 截断到 T_actual
    """
    actual_seq_q = _actual_seq_q()
    actual_seq_kv = _actual_seq_kv()
    if actual_seq_q is None:
        actual_seq_q = [0]
    if actual_seq_kv is None:
        actual_seq_kv = [0]

    inputs = prepare_npu_inputs_gqa_fp8(
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        block_table_torch=block_table_torch,
    )

    logger.info(
        "[NPU GQA FP8] 调用 %s 模式 (GRAPH_PATH=%d)...",
        "PA",
        GRAPH_PATH,
    )

    atten_out, lse_out = fa_run_npu(
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
        cu_seqlens_q_t=cu_seqlens_q_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
    )

    # T_actual: 以 actual_seq_q (seqused_q 优先, 否则从 cu_seqlens_q 差分) 为准,
    # 与 golden (assets/impl/golden.py cpu_qfa_gqa_fp8) 的截断逻辑保持一致.
    T_actual = (
        sum(actual_seq_q)
        if actual_seq_q
        else (
            cu_seqlens_q[-1]
            if cu_seqlens_q is not None and len(cu_seqlens_q) > 1
            else 0
        )
    )
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    logger.info("[NPU GQA FP8] output=%s", atten_out.shape)
    return atten_out, lse_out
