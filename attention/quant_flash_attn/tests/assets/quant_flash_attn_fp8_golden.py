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
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# ==============================================================================
# FP8 量化 (per-token-head Q/K, per-head V, descale=FP32)
# ==============================================================================
FP8_DTYPE = torch.float8_e4m3fn
K_SCALE_ROWS = 4
EPSILON = 1e-20
Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256
NUM_BLOCKS = 0


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    """GQA broadcast: [B, N_kv, S, D] → [B, N_q, S, D] (N_q = N_kv * factor)."""
    if num_heads == num_kv_heads:
        return kv_tensor.contiguous()
    factor = num_heads // num_kv_heads
    return kv_tensor.repeat_interleave(factor, dim=1).contiguous()


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
