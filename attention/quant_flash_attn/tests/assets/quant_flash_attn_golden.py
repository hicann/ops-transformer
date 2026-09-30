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
MXFP8 Flash Attention Golden

功能：生成 BNSD 数据 → CPU golden 计算 → layout 转换 → 精度对比
支持：PA / 非PA 场景，GQA
量化：Q/K per-token-group (quant_mode=6), V per-channel-group (quant_mode=8)
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import logging
import math
from typing import Optional
import torch
import torch_npu
import os as _os

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# metadata 算子 (SectionStreamK 分核信息), 用于 section 分段 golden;
# 导入失败时 golden 退化为全局 online softmax 语义
try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata

    _HAS_NPU = True
except ImportError as e:
    logger.warning("Failed to import quant_flash_attn_metadata: %s", e)
    quant_flash_attn_metadata = None
    _HAS_NPU = False

# ==============================================================================
# 配置区
# ==============================================================================
E8M0_MIN_POSITIVE = 2 ** (-127)
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 32

# ==============================================================================
# 量化 scale 计算
# MXFP8 量化算法:
#   Q/K: per-token-group, 沿 D 维度按 group_size 分组，每组独立计算 shared exponent
#   V:   per-channel-group, 沿 S 维度按 group_size 分组，每组独立计算 shared exponent
#   quant_scale = 2^(floor(log2(max_abs)) - emax)，全零组 scale=1
#   量化: quantized = original / quant_scale
#   反量化: dequantized = quantized * dequant_scale, 其中 dequant_scale = quant_scale
# ==============================================================================
_EMAX_MAP = {
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 15,
}


def _validate_fp8_dtype(fp8_dtype):
    if fp8_dtype not in _EMAX_MAP:
        raise ValueError(
            f"{fp8_dtype} not supported, expected one of {list(_EMAX_MAP.keys())}"
        )


def get_mxfp8_per_token_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized Q/K per-token-group quant_scale."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]

    dim1, dim2, dim3, dim4 = tensor.shape
    dim4_align = (dim4 + 63) // 64 * 64
    num_groups = math.ceil(dim4_align / group_size)
    pad_size = num_groups * group_size - dim4
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    grouped = tensor.reshape(dim1, dim2, dim3, num_groups, group_size)
    all_zero_mask = torch.all(grouped == 0, dim=-1)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def get_mxfp8_per_channel_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized V per-channel-group quant_scale."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]

    dim1, dim2, dim3, dim4 = tensor.shape
    num_groups = math.ceil(dim3 / group_size)
    pad_size = num_groups * group_size - dim3
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))

    grouped = tensor.reshape(dim1, dim2, num_groups, group_size, dim4)
    all_zero_mask = torch.all(grouped == 0, dim=-2)
    max_vals = torch.max(torch.abs(grouped), dim=-2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def mxfp8_per_token_group_quant(tensor, quant_scale, group_size=32):
    dim4 = tensor.shape[-1]
    scale_expanded = quant_scale.repeat_interleave(group_size, dim=-1)[..., :dim4]
    return (tensor / scale_expanded.to(tensor.dtype)).to(torch.float32)


def mxfp8_per_channel_group_quant(tensor, quant_scale, group_size=32):
    dim3 = tensor.shape[2]
    scale_expanded = quant_scale.repeat_interleave(group_size, dim=2)[:, :, :dim3, :]
    return (tensor / scale_expanded.to(tensor.dtype)).to(torch.float32)


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D = kv_tensor.shape
    result = torch.zeros([B, num_heads, S, D], dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, :, :] = kv_tensor[:, i // factor : i // factor + 1, :, :]
    return result


# ==============================================================================
# Layout 转换函数 - 数据 (Q/K/V)
# ==============================================================================


def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    """Q/K/V BNSD → 各种 layout，支持 fp8 tensor
    cu_seqlens: TND layout 时用于偏移量放置，T=cu_seqlens[-1]；None 时按 seq_lens 顺序紧凑排列
    """
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return (
            tensor[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * D)
            .contiguous()
        )
    elif layout == "TND":
        if cu_seqlens is not None:
            T = cu_seqlens[-1]
            result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
            for b in range(B):
                act_s = seq_lens[b]
                offset = cu_seqlens[b]
                if act_s <= 0:
                    continue
                for n in range(N):
                    result[offset : offset + act_s, n, :] = tensor[b, n, :act_s, :]
            return result.contiguous()
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t : t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_kv_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    return convert_q_bnsd_to_layout(
        tensor_bnsd, seq_lens, layout, cu_seqlens=cu_seqlens
    )


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    """TND layout 中 cu_seqlens padding 区域（seqused[b] ~ cu_diff[b]）填充指定值
    用于 LSE: padding 位置填 inf 以匹配 NPU 行为
    """
    if cu_seqlens is None:
        return tensor_tnd
    B = len(seq_lens)
    for b in range(B):
        act_s = seq_lens[b]
        offset = cu_seqlens[b]
        cu_diff = cu_seqlens[b + 1] - cu_seqlens[b]
        if cu_diff > act_s:
            tensor_tnd[offset + act_s : offset + cu_diff] = fill_value
    return tensor_tnd


# ==============================================================================
# Layout 转换函数 - Scale (Q/K/V)
# ==============================================================================


def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 → e8m0fnu，提取 IEEE 754 biased exponent
    e8m0fnu 格式: 只有指数位，没有尾数，表示 2^(e-127)
    biased exponent = 0xFF 时表示 NaN
    返回 torch.float8_e8m0fnu 以匹配 def.cpp 中 DT_FLOAT8_E8M0 的 dtype 定义
    """
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp.view(torch.float8_e8m0fnu)


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0fnu 没有 0 值语义；非有限值进入 0xFF 会在 NPU 侧变 NaN。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad_mask = ~torch.isfinite(result)
    zero_mask = result == 0
    bad_count = int(bad_mask.sum().item())
    zero_count = int(zero_mask.sum().item())
    if bad_count:
        logger.info(
            "[WARN] %s: replace %d non-finite scale values before e8m0 packing",
            name,
            bad_count,
        )
        result[bad_mask] = E8M0_MIN_POSITIVE
    if zero_count:
        result[zero_mask] = E8M0_MIN_POSITIVE
    return result


def fp32_to_e8m0fnu_safe(scale, name="scale"):
    scale_safe = sanitize_e8m0_scale(scale, name)
    packed = fp32_to_e8m0fnu(scale_safe)
    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(
            f"{name}: {nan_byte_count} values would become e8m0fnu NaN (0xFF)"
        )
    return packed


def e8m0_to_fp32(tensor_e8m0):
    biased_exp = tensor_e8m0.view(torch.uint8).to(torch.float32)
    result = torch.pow(2.0, biased_exp - 127)
    # biased_exp == 0xFF is the NaN sentinel in e8m0fnu — force to zero
    nan_mask = biased_exp == 0xFF
    if nan_mask.any():
        result = result.clone()
        result[nan_mask] = 0.0
    return result


def canonical_q_scale_layout(layout):
    layout = (layout or "TND").upper()
    if layout not in ("TND", "N2TGD"):
        raise ValueError(f"Unsupported Q scale layout: {layout}")
    return layout


def _convert_q_scale_bnsd_to_tnd(scale_bnsd, seq_lens, cu_seqlens=None):
    B, N, _, Dg = scale_bnsd.shape
    Dg_half = Dg // 2
    if cu_seqlens is not None:
        T = cu_seqlens[-1]
        result = torch.zeros(
            (T, N, Dg_half, 2), dtype=scale_bnsd.dtype, device=scale_bnsd.device
        )
        for b in range(B):
            act_s = seq_lens[b]
            offset = cu_seqlens[b]
            if act_s <= 0:
                continue
            for n in range(N):
                result[offset : offset + act_s, n, :, :] = scale_bnsd[
                    b, n, :act_s, :
                ].reshape(act_s, Dg_half, 2)
        return result
    T = sum(seq_lens)
    result = torch.zeros(
        (T, N, Dg_half, 2), dtype=scale_bnsd.dtype, device=scale_bnsd.device
    )
    t = 0
    for b in range(B):
        act_s = seq_lens[b]
        for n in range(N):
            result[t : t + act_s, n, :, :] = scale_bnsd[b, n, :act_s, :].reshape(
                act_s, Dg_half, 2
            )
        t += act_s
    return result


def convert_q_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, cu_seqlens=None):
    """Q scale BNSD → 各种 layout (已 packed: D//2, 2)"""
    layout = canonical_q_scale_layout(layout)
    B, N, _, Dg = scale_bnsd.shape
    max_org_s = max(seq_lens)
    Dg_half = Dg // 2

    if layout == "BNSD":
        return scale_bnsd[:, :, :max_org_s, :].reshape(B, N, max_org_s, Dg_half, 2)
    elif layout == "BSND":
        return (
            scale_bnsd[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N, Dg_half, 2)
        )
    elif layout == "BSH":
        return (
            scale_bnsd[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * Dg_half, 2)
        )
    elif layout == "TND":
        return _convert_q_scale_bnsd_to_tnd(scale_bnsd, seq_lens, cu_seqlens=cu_seqlens)
    elif layout == "N2TGD":
        tnd_result = _convert_q_scale_bnsd_to_tnd(
            scale_bnsd, seq_lens, cu_seqlens=cu_seqlens
        )
        return convert_q_scale_tnd_to_n2tgd_layout(tnd_result, N_kv)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_k_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, cu_seqlens=None):
    return convert_q_scale_bnsd_to_layout(
        scale_bnsd, seq_lens, layout, cu_seqlens=cu_seqlens
    )


def convert_v_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, group_size=32):
    """V scale BNSD → 各种 layout
    V scale 偶奇行交错 packing: result[..., 0] = 偶数行, result[..., 1] = 奇数行
    奇数 Sg 时 pad 一行 E8M0_MIN_POSITIVE
    """
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    if actual_Sg % 2 != 0:
        actual_Sg_padded = actual_Sg + 1
    else:
        actual_Sg_padded = actual_Sg

    S_out = actual_Sg_padded // 2

    if layout == "BNSD":
        transposed = scale_bnsd[:, :, :actual_Sg, :]
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, N, 1, D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=2)
        result = torch.zeros(
            (B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[..., ::2, :]
        result[..., 1] = transposed[..., 1::2, :]
        return result
    elif layout == "BSND":
        transposed = scale_bnsd[:, :, :actual_Sg, :].permute(0, 2, 1, 3)
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, 1, N, D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros(
            (B, S_out, N, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[:, ::2, :, :]
        result[..., 1] = transposed[:, 1::2, :, :]
        return result
    elif layout == "BSH":
        transposed = (
            scale_bnsd[:, :, :actual_Sg, :]
            .permute(0, 2, 1, 3)
            .reshape(B, actual_Sg, N * D)
        )
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, 1, N * D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros(
            (B, S_out, N * D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[:, ::2, :]
        result[..., 1] = transposed[:, 1::2, :]
        return result
    elif layout == "TND":
        Tv = 0
        for seq_len in seq_lens:
            sg = math.ceil(seq_len / group_size)
            sg_padded = sg + (sg % 2)
            Tv += sg_padded // 2

        result = torch.zeros(
            (Tv, N, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        t_start = 0
        for b in range(B):
            org_seq = seq_lens[b]
            sg = math.ceil(org_seq / group_size)
            sg_padded = sg + (sg % 2)
            act_s = sg_padded // 2
            t_end = t_start + act_s
            if act_s <= 0:
                continue
            for n in range(N):
                src = scale_bnsd[b, n, :sg, :]
                if sg % 2 != 0:
                    pad = torch.full(
                        (1, D), E8M0_MIN_POSITIVE, dtype=src.dtype, device=src.device
                    )
                    src = torch.cat([src, pad], dim=0)
                result[t_start:t_end, n, :, 0] = src[::2, :]
                result[t_start:t_end, n, :, 1] = src[1::2, :]
            t_start = t_end
        return result
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_q_scale_tnd_to_n2tgd_layout(tensor_tnd, num_kv_heads):
    """
    输入: (T, N_q, D//2, 2)
    输出: (N_kv, T, G, D//2, 2), G = N_q / N_kv
    N2TGD: N_kv 组，每组 G 个 query head 共享同一个 kv head
    """
    T, N, D, _ = tensor_tnd.shape
    G = N // num_kv_heads
    tensor_reshape = tensor_tnd.reshape(T, num_kv_heads, G, D, 2)
    return tensor_reshape.permute(1, 0, 2, 3, 4).contiguous()


# ==============================================================================
# TND → BNSD 反向转换 + scale unpack 辅助函数
# ==============================================================================


# ==============================================================================
# PA 格式转换 - mxfp8_pa_preprocessing
# ==============================================================================


def mxfp8_pa_preprocessing(
    tensor_bnsd,
    seq_lens,
    block_size,
    block_table,
    is_vscale=False,
    is_scale=False,
    kv_layout="BnNBsD",
    group_size=32,
    *,
    total_blocks: Optional[int] = None,
):
    """
    MXFP8 PA 预处理: BNSD → PagedAttention KV Cache

    输入: [B, N, S, D]
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K scale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V scale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], Kscale=[Bn,N,Bs,D//64,2], Vscale=[Bn,N,Bs//64,D,2]
      - PA_BNBD: 同 BnNBsD, [Bn,N,Bs,D]
      - BnBsND: fp8=[Bn,Bs,N,D], Kscale=[Bn,Bs,N,D//64,2], Vscale=[Bn,Bs//64,N,D,2]
      - PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
    """
    tensor_bnsd = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    B, N, S, D = tensor_bnsd.shape

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_scale_pack_ratio = group_size * 2
            pack_seq_lens = [
                math.ceil(act_s / v_scale_pack_ratio) for act_s in seq_lens
            ]
            pack_block_size = math.ceil(block_size / v_scale_pack_ratio)
            total_block_num = (
                total_blocks
                if total_blocks is not None
                else sum(math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens)
            )
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K scale D must be even for packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (
                total_blocks
                if total_blocks is not None
                else sum(math.ceil(act_s / block_size) for act_s in seq_lens),
                N,
                block_size,
                D // 2,
                2,
            )
    else:
        tensor_processed = tensor_bnsd
        pack_seq_lens = seq_lens
        pack_block_size = block_size
        out_shape = (
            total_blocks
            if total_blocks is not None
            else sum(math.ceil(act_s / block_size) for act_s in seq_lens),
            N,
            block_size,
            D,
        )

    fill_value = E8M0_MIN_POSITIVE if is_scale else 0

    out_cache = torch.full(
        out_shape,
        fill_value,
        dtype=tensor_processed.dtype,
        device=tensor_processed.device,
    )
    block_num = [math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_block_size
            valid_len = min(pack_block_size, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[blockid, :, :valid_len] = tensor_processed[
                b, :, block_offset : block_offset + valid_len
            ]
    if kv_layout in ("BnNBsD", "PA_BNBD"):
        # PA_BNBD: [Bn, N, Bs, D] — 和 BnNBsD 维度排列一致
        return out_cache
    elif kv_layout in ("BnBsND", "PA_BBND"):
        return out_cache.transpose(1, 2).contiguous()
    elif kv_layout == "PA_NZ":
        Bn, KV_N, Bs, KV_D = out_cache.shape[:4]
        if not is_scale:
            inner = 32
            if KV_D % inner != 0:
                raise ValueError(f"PA_NZ D must be divisible by {inner}, got {KV_D}")
            reshaped = out_cache.reshape(Bn, KV_N, Bs, KV_D // inner, inner)
            return reshaped.permute(0, 1, 3, 2, 4).contiguous()
        elif not is_vscale:
            reshaped = out_cache.reshape(Bn, KV_N, Bs // 16, 16, KV_D, 2)
            return reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
        else:
            reshaped = out_cache.reshape(Bn, KV_N, Bs, KV_D // 16, 16, 2)
            return reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def convert_v_scale_to_pa(scale_bnsd, seq_lens, group_size=32):
    """
    V scale 偶奇行交错 packing，用于 PA 预处理
    输入: [B, N, Sg, D], Sg = ceil(orgS/32)
    输出: [B, N, Sg//2, D, 2] (奇数行 pad 到偶数)
    """
    scale_bnsd = (
        scale_bnsd
        if isinstance(scale_bnsd, torch.Tensor)
        else torch.as_tensor(scale_bnsd)
    )
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    transposed = scale_bnsd[:, :, :actual_Sg, :]

    if actual_Sg % 2 != 0:
        pad = torch.full(
            (B, N, 1, D),
            E8M0_MIN_POSITIVE,
            dtype=transposed.dtype,
            device=transposed.device,
        )
        transposed = torch.cat([transposed, pad], dim=2)
        actual_Sg += 1

    S_out = actual_Sg // 2
    result = torch.zeros(
        (B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device
    )
    result[..., 0] = transposed[..., ::2, :]
    result[..., 1] = transposed[..., 1::2, :]
    return result


# ==============================================================================
# PA 格式逆转换 - PA → BNSD reverse
# ==============================================================================


def pa_reverse_permute_nz(tensor, is_scale, is_vscale):
    tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    if not is_scale:
        # fp8 data: [Bn,N,D//32,Bs,32] → permute(0,1,3,2,4) → [Bn,N,Bs,D//32,32]
        # then reshape → [Bn,N,Bs,D]
        t = tensor.permute(0, 1, 3, 2, 4).contiguous()
        Bn, N, Bs, inner, tail = t.shape
        D = inner * tail
        return t.reshape(Bn, N, Bs, D)
    elif not is_vscale:
        # K scale: [Bn,N,Bs//16,D//2,16,2] → permute(0,1,2,4,3,5) → [Bn,N,Bs//16,16,D//2,2]
        # then reshape → [Bn,N,Bs,D//2,2]
        t = tensor.permute(0, 1, 2, 4, 3, 5).contiguous()
        Bn, N, Bs_div_16, inner, D_half, _pair = t.shape
        Bs = Bs_div_16 * inner
        return t.reshape(Bn, N, Bs, D_half, 2)
    else:
        # V scale: [Bn,N,D//16,Bs,16,2] → permute(0,1,3,2,4,5) → [Bn,N,Bs,D//16,16,2]
        # then reshape → [Bn,N,Bs,D,2]
        t = tensor.permute(0, 1, 3, 2, 4, 5).contiguous()
        Bn, N, Bs, D_div_16, inner, tail2 = t.shape
        D = D_div_16 * inner
        return t.reshape(Bn, N, Bs, D, 2)


def pa_to_bnsd_data(pa_tensor, seq_lens, block_size, block_table, kv_layout="BnNBsD"):
    pa_tensor = (
        pa_tensor if isinstance(pa_tensor, torch.Tensor) else torch.as_tensor(pa_tensor)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    Bn, N, Bs, D = pa_tensor.shape
    B = block_table.shape[0]
    max_skv = max(seq_lens)

    result = torch.zeros(
        B,
        N,
        max_skv,
        D,
        dtype=pa_tensor.dtype,
        device=pa_tensor.device,
    )
    num_blocks = [math.ceil(s / block_size) for s in seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(num_blocks[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            result[b, :, block_offset : block_offset + valid_len] = pa_tensor[
                blockid, :, :valid_len
            ]
    return result[:, :, : max(seq_lens), :].contiguous()


# ==============================================================================
# 数据生成
# ==============================================================================


# ==============================================================================
# CPU Golden
# Flash Attention tiling 策略: C1V1C1V1C2V2 流水
#   Q 按 Q_BLOCK_SIZE 分块, K/V 按 K_BLOCK_SIZE/V_BLOCK_SIZE 分块
#   每次迭代处理 2 个 K block (j, j+1) 和 1 个 V block
#   Online softmax: 维护 running max (m) 和 running sum (s) 实现数值稳定
#   TND layout 下 m 需要对齐到 ln2 的整数倍 (ceil)
# ==============================================================================


def _build_attention_mask(b, Sq, Skv, actual_seq_q, actual_seq_kv, sparse_mode):
    """构建全局 attention mask
    sparse_mode=3: causal + padding mask (左下三角 + 右上 padding)
    其他: 仅 padding mask
    """
    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(b, 1, 1, 1)
    k_lens_acl = k_lens_t.view(b, 1, 1, 1)

    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if sparse_mode == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        return causal_mask | q_padding_mask | k_padding_mask
    else:
        return q_padding_mask | k_padding_mask


def _compute_s_block(Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale):
    """计算单个 S block (attention score)"""
    S_ij = torch.matmul(Qi * deq_scale_q_i, (Kj * deq_scale_k_j).permute(0, 1, 3, 2))
    return S_ij * softmax_scale


def _online_softmax_update(S_ij, mask_j, mi, si, oi, ln_p_scale):
    """Online softmax: 计算 m, P, s 更新 (MXFP8: stored max 不含 -ln(p_scale), P 含 p_scale 因子)
    1. mask 位置填 -inf
    2. 求 block 内 max (m_block_j)
    3. m 对齐到 ln2 整数倍 (ceil)，模拟 NPU MXFP8 量化精度损失
    4. 与前一个 block 的 m 取 max (stored max 不含 -ln(p_scale))
    5. 计算 P = exp(S - m + ln_p_scale)，模拟 NPU: exp 用 adjusted max (含 -ln(p_scale))，但 stored max 不含
    6. 求 s = sum(P)
    7. P 转 FP8 再转回 FP32，模拟 NPU 侧 P 的量化损失
    """
    LN2 = 0.6931471824645996
    INV_LN2 = 1.4426950216293335
    S_ij = S_ij.masked_fill(mask_j, float("-inf"))

    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
    m_block_j = torch.ceil(m_block_j * INV_LN2) * LN2
    m_block_j = m_block_j - ln_p_scale
    m_block_j = torch.max(mi, m_block_j)

    P_ij_raw = torch.exp(S_ij - m_block_j)
    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
    P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)

    return m_block_j, s_block_j, P_ij_drop


# ==============================================================================
# Kernel SectionStreamK 分核调度解码
# NPU kernel 将 (bn, m-block, s2-block) 线性迭代空间按 core 切分为多个 chunk,
# metadata 的 FA 区记录每个 (section, core) 的 chunk [start, end)。
# chunk 内 online softmax 连续, 跨 chunk 的结果在 FD 阶段按 max 重标定合并。
# causal (mask_mode=3) 时每个 m 的 s2 块范围由 CalcS2Range 决定。
# golden 需按相同分段模拟才能对齐 P 的 fp8 量化网格 (分段越长量化点越稀)。
# ==============================================================================
_QFA_SECTION_CACHE = {}


def resolve_q_scale_layout(layout=None):
    """解析 Q scale layout 并返回 (resolved_layout, gqa_group)"""
    resolved = canonical_q_scale_layout(layout or globals().get("Q_SCALE_LAYOUT"))
    if N_kv <= 0 or N_q % N_kv != 0:
        raise ValueError(f"N_q must be divisible by N_kv, got N_q={N_q}, N_kv={N_kv}")
    group = N_q // N_kv
    return resolved, group


def _qfa_calc_s2_block_range(mi, m_base, s2_base, q_len, kv_len, sched_group):
    """复刻 kernel CalcS2Range (section_stream_k_impl.h L560-624)

    m 轴为 S1G 合轴行 (row = s1 * sched_group + g; sched_group=1 时即 s1)。
    支持 mask_mode=3 RIGHT_DOWN_CAUSAL (preToken=querySeq, nextToken=kvS-qS,
    见 base_info.h GetPreTokenLeftUp/GetNextTokenLeftUp) 与 mask_mode=0/5
    (无 mask 全范围); 其它模式不支持, 由调用方回退全局语义。
    返回 m-block mi 的 (s2Start块, s2End块), s2End 为开区间; 无有效范围返回 (0, 0)。
    """
    if q_len == 0 or kv_len == 0:
        return 0, 0
    if SPARSE_MODE in (0, 5):
        return 0, (kv_len + s2_base - 1) // s2_base
    if SPARSE_MODE != 3:
        raise ValueError(f"golden 分段解码不支持 mask_mode={SPARSE_MODE}")
    m_size = q_len * sched_group
    m_first = mi * m_base
    if m_first >= m_size:
        return 0, 0
    m_last = min(m_first + m_base, m_size) - 1  # NumToIndex
    s1_first = m_first // sched_group  # GetIsS1G (TND/BSH/BSND) 合轴映射
    s1_last = m_last // sched_group  # closed
    s2_first_tok = s1_first - q_len  # preTokenLeftUp = querySeq, 恒 <= 0
    s2_last_tok = s1_last + kv_len - q_len  # nextTokenLeftUp = kvS - qS
    if s2_first_tok >= kv_len or s2_last_tok < 0 or s2_last_tok < s2_first_tok:
        return 0, 0
    s2_first_tok = max(0, min(s2_first_tok, kv_len - 1))  # Clip(0, NumToIndex(s2Size))
    s2_last_tok = max(0, min(s2_last_tok, kv_len - 1))
    return s2_first_tok // s2_base, s2_last_tok // s2_base + 1  # ToOpenInterval = +1


def get_qfa_section_info(actual_seq_q, actual_seq_kv):
    """调用 metadata 算子并解码 FA 分核分段信息

    返回 (m_base, s2_base, segments, head_num, is_decode):
      segments[bn][mi] = [(s2块起, s2块止), ...] 该 (bn, m-block) 的分段列表
    解码失败时返回 None, golden 退化为全局 online softmax 语义。
    """
    key = (
        tuple(int(x) for x in actual_seq_q),
        tuple(int(x) for x in actual_seq_kv),
        N_q,
        N_kv,
        D,
        SPARSE_MODE,
        globals().get("INPUT_LAYOUT"),
        globals().get("KV_CACHE_LAYOUT"),
        globals().get("ENABLE_PA"),
        globals().get("MAX_SEQLEN_Q"),
        globals().get("MAX_SEQLEN_KV"),
        globals().get("Q_SCALE_LAYOUT"),
    )
    if key in _QFA_SECTION_CACHE:
        return _QFA_SECTION_CACHE[key]
    info = None
    if _HAS_NPU:
        try:
            info = _build_qfa_section_info(actual_seq_q, actual_seq_kv)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[Section] metadata 调用/解码失败(%s), golden 使用全局量化语义", exc
            )
    _QFA_SECTION_CACHE[key] = info
    return info


def _build_qfa_section_info(actual_seq_q, actual_seq_kv):
    """调用 metadata 算子 (参数与 NPU 侧主算子保持一致) 并解码分段"""
    if quant_flash_attn_metadata is None:
        return None
    torch.npu.set_device(int(globals().get("DEVICE_ID", 0)))

    enable_pa = bool(globals().get("ENABLE_PA", False))
    input_layout = globals().get("INPUT_LAYOUT") or "TND"
    kv_cache_layout = globals().get("KV_CACHE_LAYOUT") or "BnNBsD"
    layout_q = "TND" if enable_pa else input_layout
    _pa_layout_kv_map = {"BnNBsD": "PA_BNBD", "BnBsND": "PA_BBND", "PA_NZ": "PA_NZ"}
    layout_kv = (
        _pa_layout_kv_map.get(kv_cache_layout, "PA_BNBD") if enable_pa else input_layout
    )
    layout_out = "TND" if enable_pa else input_layout
    q_runtime_layout, _ = resolve_q_scale_layout()

    def _derive_cu(seqused):
        cu = [0]
        acc = 0
        for seq in seqused:
            acc += int(seq)
            cu.append(acc)
        return cu

    cu_q = globals().get("CU_SEQLENS_Q")
    cu_q = cu_q if cu_q is not None else _derive_cu(actual_seq_q)
    cu_kv = globals().get("CU_SEQLENS_KV")
    cu_kv = cu_kv if cu_kv is not None else _derive_cu(actual_seq_kv)

    seqused_q_t = torch.tensor(actual_seq_q, dtype=torch.int32).npu().contiguous()
    seqused_kv_t = torch.tensor(actual_seq_kv, dtype=torch.int32).npu().contiguous()
    cu_q_t = (
        torch.tensor(cu_q, dtype=torch.int32).npu().contiguous()
        if layout_q == "TND"
        else None
    )
    cu_kv_t = (
        torch.tensor(cu_kv, dtype=torch.int32).npu().contiguous()
        if layout_kv == "TND"
        else None
    )

    max_sq = globals().get("MAX_SEQLEN_Q")
    max_sq = (
        max_sq
        if (max_sq is not None and max_sq > 0)
        else max(int(x) for x in actual_seq_q)
    )
    max_skv = globals().get("MAX_SEQLEN_KV")
    max_skv = (
        max_skv
        if (max_skv is not None and max_skv > 0)
        else max(int(x) for x in actual_seq_kv)
    )

    meta = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=1,
        cu_seqlens_q=cu_q_t,
        cu_seqlens_kv=cu_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        mask_mode=SPARSE_MODE,
        layout_q=layout_q,
        layout_q_descale=q_runtime_layout,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
    )
    torch.npu.synchronize()
    meta_flat = meta.contiguous().cpu().view(-1).tolist()
    return _decode_qfa_sections(meta_flat, actual_seq_q, actual_seq_kv)


def _decode_qfa_sections(meta_flat, seqused_q, seqused_kv):
    """解码 metadata: header[16] + FA区[section][aicNum][16] (+FD区, 不需要)

    每个 FA chunk 是 (bn, m, s2) 线性迭代空间上的一段 [start, end)。
    bn 语义 (aicpu quant_flash_attn_metadata_aicpu.cpp L271-316):
      - 非 decode (layout_q_descale != "N2TGD"): kvHeadNum=numHeadsQ_ →
        headNum=N_q, bn = b*N_q + query_head; kernel USE_DN (prefill 不合轴)
        realN2Size=n2Size*gSize=N_q, realGSize=1 → m 行即 s1 token
      - decode: headNum=N_kv, bn = b*N_kv + kv_head; kernel 合轴
        realGSize=gSize → m 行 = s1*G + g
    sched_group = GetGroupSize() = N_q/headNum (prefill=1, decode=G)。
    causal 时每个 m 的 s2 块范围由 CalcS2Range 决定。chunk 即 core 分段,
    chunk 内 running max 连续。
    """
    META_SIZE = 16
    section_num = int(meta_flat[0])
    is_fd = int(meta_flat[1])
    m_base = int(meta_flat[2])
    s2_base = int(meta_flat[3])
    aic_num = int(meta_flat[4])
    if section_num <= 0 or m_base <= 0 or s2_base <= 0 or aic_num <= 0:
        raise ValueError(
            f"invalid metadata header: section={section_num}, mBase={m_base}, "
            f"s2Base={s2_base}, aicNum={aic_num}"
        )

    is_decode = resolve_q_scale_layout()[0] == "N2TGD"
    head_num = N_kv if is_decode else N_q
    sched_group = N_q // head_num

    batch = len(seqused_kv)
    bn_total = batch * head_num
    m_blocks = [
        (int(seqused_q[bi]) * sched_group + m_base - 1) // m_base for bi in range(batch)
    ]
    # 每个 (bn, mi) 的 s2 块范围 [s2Start, s2End)
    s2_ranges = []
    for bi in range(batch):
        q_len = int(seqused_q[bi])
        kv_len = int(seqused_kv[bi])
        s2_ranges.append(
            [
                _qfa_calc_s2_block_range(
                    mi, m_base, s2_base, q_len, kv_len, sched_group
                )
                for mi in range(m_blocks[bi])
            ]
        )

    segments = [[[] for _ in range(m_blocks[bn // head_num])] for bn in range(bn_total)]

    def _advance(bn, mi):
        mi += 1
        if mi >= m_blocks[bn // head_num]:
            mi = 0
            bn += 1
        return bn, mi

    for sec in range(section_num):
        for core in range(aic_num):
            base_idx = META_SIZE + (sec * aic_num + core) * META_SIZE
            bn_s, m_s, s2_s, bn_e, m_e, s2_e = (
                int(v) for v in meta_flat[base_idx : base_idx + 6]
            )
            if bn_s == bn_e and m_s == m_e and s2_s == s2_e:
                continue  # 未使用的核 (全 0) 或空 chunk
            bn, mi, s2 = bn_s, m_s, s2_s
            steps = 0
            while (bn, mi, s2) != (bn_e, m_e, s2_e):
                steps += 1
                if steps > 100000000 or bn >= bn_total:
                    raise ValueError("metadata chunk walk 越界")
                if mi >= m_blocks[bn // head_num]:
                    bn, mi = _advance(bn, mi)
                    s2 = 0  # 支持的 sparse mode 下每个 m 的 s2Start 恒为 0
                    continue
                s2_start, s2_end = s2_ranges[bn // head_num][mi]
                if s2_end <= s2_start or s2 >= s2_end:
                    bn, mi = _advance(bn, mi)
                    s2 = 0
                    continue
                seg_end = min(s2_e, s2_end) if (bn == bn_e and mi == m_e) else s2_end
                if seg_end > s2:
                    segments[bn][mi].append((s2, seg_end))
                if seg_end >= s2_end:
                    bn, mi = _advance(bn, mi)
                    s2 = 0
                else:
                    s2 = seg_end

    # 覆盖率校验: 每个 m-block 的分段必须连续且完整覆盖其 causal s2 范围
    for bi in range(batch):
        for h in range(head_num):
            bn = bi * head_num + h
            for mi in range(m_blocks[bi]):
                s2_start, s2_end = s2_ranges[bi][mi]
                cur = s2_start
                for aa, bb in segments[bn][mi]:
                    if aa != cur:
                        raise ValueError(
                            f"bn={bn} m={mi} 分段不连续: expect {cur}, got {aa}"
                        )
                    cur = bb
                if cur != s2_end:
                    raise ValueError(
                        f"bn={bn} m={mi} 分段覆盖 {cur} != {s2_end} (start={s2_start})"
                    )

    split_num = sum(1 for bn_segs in segments for segs in bn_segs if len(segs) > 1)
    logger.info(
        "[Section] 解码成功: isFd=%d, sectionNum=%d, mBase=%d, s2Base=%d, "
        "aicNum=%d, headNum=%d, schedGroup=%d, 跨核分段 (bn,m) 数=%d",
        is_fd,
        section_num,
        m_base,
        s2_base,
        aic_num,
        head_num,
        sched_group,
        split_num,
    )
    return m_base, s2_base, segments, head_num, is_decode


def _cpu_mxfp8_overflow_matmul(lhs, rhs):
    """CPU reference for overflow-risk MXFP8 products (rhs is transposed).

    Accumulate each 32-element quantization group into the FP32 accumulator.
    Widen the group's product and addition to avoid separately overflowing
    dequantized operands; round back after every group, not after the full K.
    In particular, an accumulator that overflowed to infinity stays infinity
    when later finite groups have the opposite sign.
    """
    acc = torch.zeros((lhs.shape[0], rhs.shape[1]), dtype=torch.float32)
    for start in range(0, lhs.shape[1], QUANT_GROUP_SIZE):
        end = start + QUANT_GROUP_SIZE
        group = torch.matmul(lhs[:, start:end].double(), rhs[start:end].double())
        acc = (acc.double() + group).float()
    return acc


def _cpu_mxfp8_golden_sectioned(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    section_info,
    softmax_scale=None,
):
    """分段 golden: 按 kernel SectionStreamK 分核调度模拟 online softmax 与 P 量化

    每个 (bn, m-block) 的各分段 (core chunk) 独立维护 running max, 分段结果按
    FD 语义 (max 重标定) 合并, 与 NPU 一致。
    P 的 fp8 cast 网格 = max(本块对齐 max, 段内 running max): 先在 own 网格上
    exp, 再用精确 2 的幂 (e8m0 pScale) 缩放到公共网格后 cast (小 P 进
    subnormal/flush 区), 分母累加 raw 和。
    bn 语义: prefill bn = b*N_q + query_head (sched_group=1, m 行即 s1);
    decode bn = b*N_kv + kv_head (sched_group=G, m 行 = s1*G + g)。
    """
    # 与 kernel 常量一致 (vf_basic_block_utils.h), 同 _online_softmax_update
    LN2 = 0.6931471824645996
    INV_LN2 = 1.4426950216293335
    if D == 256:
        K_BLOCK_SIZE = 128
    else:
        K_BLOCK_SIZE = 256

    m_base, s2_base, segments, head_num, is_decode = section_info
    if s2_base % K_BLOCK_SIZE != 0:
        raise ValueError(f"s2Base({s2_base}) 与 golden K block({K_BLOCK_SIZE}) 不对齐")
    chunks_per_s2 = s2_base // K_BLOCK_SIZE

    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    b, n, s, d = q_tensor.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    dv = v_tensor.shape[-1]
    sched_group = n // head_num  # GetGroupSize: prefill=1, decode=G
    gqa_group = n // N_kv  # GQA: 每 kv head 对应的 query head 数
    minValue = -3.402823466e38
    # Kernel 0xFF7FFFFE: non-DN Decode uses the unscaled sentinel;
    # DN Prefill applies softmax_scale after the raw max reduction.
    neg_min_ln2 = torch.tensor(-8388610, dtype=torch.int32).view(torch.float32).item()
    mask_fill = neg_min_ln2 if is_decode else neg_min_ln2 * softmax_scale
    tail_align = 64 if is_decode else K_BLOCK_SIZE

    # dequant_scale 按 group_size 扩展, 用于逐元素反量化
    deq_q_exp = dequant_scale_q.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., :D]
    deq_k_exp = dequant_scale_k.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., :D]
    v_ds_exp = dequant_scale_v.repeat_interleave(QUANT_GROUP_SIZE, dim=2)

    # MXFP8 dequantization may exceed FP32 even though the FP8 value and
    # E8M0 scale are both finite. Avoid premature inf/NaN on CPU: widen
    # only overflow-risk products and retain FP32 accumulation per group.
    fp32_max = torch.finfo(torch.float32).max
    q_bound = q_tensor.abs().amax().item() * dequant_scale_q.abs().amax().item()
    k_bound = k_tensor.abs().amax().item() * dequant_scale_k.abs().amax().item()
    v_bound = v_tensor.abs().amax().item() * dequant_scale_v.abs().amax().item()
    qk_dtype = torch.float64 if q_bound * k_bound * d > fp32_max else torch.float32
    pv_dtype = (
        torch.float64 if 448.0 * v_bound * K_BLOCK_SIZE > fp32_max else torch.float32
    )

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    out = torch.zeros([b, n, s, dv], dtype=torch.float32)
    lse = torch.full([b, n, s, 1], float("inf"), dtype=torch.float32)

    for bi in range(b):
        q_len = int(actual_seq_q[bi])
        kv_len = int(actual_seq_kv[bi])
        if q_len == 0 or kv_len == 0:
            continue
        m_size = q_len * sched_group
        for h in range(head_num):
            bn = bi * head_num + h
            # prefill: h 为 query head, kv_head = h // gqa_group;
            # decode: h 为 kv head
            kv_head = h if is_decode else h // gqa_group
            k_bn = k_tensor[bi, kv_head]
            dk_bn = deq_k_exp[bi, kv_head]
            v_bn = v_tensor[bi, kv_head]
            vd_bn = v_ds_exp[bi, kv_head]
            for mi, segs in enumerate(segments[bn]):
                r0 = mi * m_base
                r1 = min((mi + 1) * m_base, m_size)
                if r1 <= r0 or not segs:
                    continue
                rows = torch.arange(r0, r1)
                s1_idx = (rows // sched_group).long()
                g_idx = (rows % sched_group).long()
                head_idx = h * sched_group + g_idx
                q_sel = q_tensor[bi, head_idx, s1_idx, :].to(qk_dtype) * deq_q_exp[
                    bi, head_idx, s1_idx, :
                ].to(qk_dtype)
                causal_bound = None
                if SPARSE_MODE == 3:
                    causal_bound = (s1_idx + (kv_len - q_len)).long()[:, None]

                o_g = torch.zeros((r1 - r0, dv), dtype=torch.float32)
                s_g = torch.zeros((r1 - r0, 1), dtype=torch.float32)
                m_g = torch.full((r1 - r0, 1), minValue, dtype=torch.float32)

                m_valid_g = torch.full((r1 - r0,), float("-inf"), dtype=torch.float32)
                segment_stats = []
                for blk_a, blk_b in segs:
                    # 每个分段独立 online softmax (对应 kernel 的一个 core chunk)
                    o = torch.zeros_like(o_g)
                    st = torch.zeros_like(s_g)
                    mx = torch.full_like(m_g, minValue)
                    for j in range(blk_a * chunks_per_s2, blk_b * chunks_per_s2):
                        lo = j * K_BLOCK_SIZE
                        hi = min(
                            lo + K_BLOCK_SIZE,
                            math.ceil(kv_len / tail_align) * tail_align,
                        )
                        if lo >= hi:
                            break
                        # Only real KV rows participate in QK/PV. Padding scores
                        # contribute to the denominator, with zero V contribution.
                        data_hi = min(hi, kv_len)
                        kj = k_bn[lo:data_hi, :].to(qk_dtype) * dk_bn[lo:data_hi, :].to(
                            qk_dtype
                        )
                        if qk_dtype == torch.float64:
                            sij = _cpu_mxfp8_overflow_matmul(q_sel, kj.transpose(0, 1))
                        else:
                            sij = torch.matmul(q_sel, kj.transpose(0, 1))
                        sij = sij * softmax_scale
                        if hi > data_hi:
                            sij = torch.nn.functional.pad(
                                sij, (0, hi - data_hi), value=mask_fill
                            )
                        cols = torch.arange(lo, hi)
                        mask_j = (cols >= kv_len)[None, :].expand(r1 - r0, hi - lo)
                        if causal_bound is not None:
                            mask_j = mask_j | (cols[None, :] > causal_bound)
                        sij_masked = sij.masked_fill(mask_j, mask_fill)
                        valid_scores = sij.masked_fill(mask_j, float("-inf"))
                        m_valid_g = torch.maximum(m_valid_g, valid_scores.amax(dim=-1))
                        m_blk_j = sij_masked.amax(dim=-1, keepdim=True)
                        if not is_decode:
                            m_blk_j = torch.clamp(m_blk_j, min=mask_fill)
                        m_own = torch.ceil(m_blk_j * INV_LN2) * LN2 - ln_p_scale
                        if is_decode:
                            m_own = torch.clamp(m_own, min=neg_min_ln2)
                        m_own_safe = torch.where(
                            m_blk_j == float("-inf"),
                            torch.zeros_like(m_own),
                            m_own,
                        )
                        s_blk = torch.sum(
                            torch.exp(sij_masked - m_own_safe), dim=-1, keepdims=True
                        )
                        # kernel 语义 (ProcessVec1DnUpdateMxfp8VF L1239-1259):
                        # P 在 max(本块对齐 max, 段内 running max) 网格上 cast
                        # fp8, 先做 2 的幂缩放再 cast (小 P 进 subnormal/flush,
                        # 与 own-grid cast 不同); 分母累加 raw 和
                        m_run_new = torch.max(mx, m_own)
                        upd_old = torch.exp(mx - m_run_new)
                        upd_old = torch.where(
                            mx <= minValue, torch.zeros_like(upd_old), upd_old
                        )
                        upd_cur = torch.exp(m_own - m_run_new)
                        p_run = torch.exp(sij_masked - m_own_safe) * upd_cur
                        p_drop = p_run.to(FP8_DTYPE).to(torch.float32)
                        vj = v_bn[lo:data_hi, :].to(pv_dtype) * vd_bn[lo:data_hi, :].to(
                            pv_dtype
                        )
                        if hi > data_hi:
                            vj = torch.nn.functional.pad(vj, (0, 0, 0, hi - data_hi))
                        if pv_dtype == torch.float64:
                            pv = _cpu_mxfp8_overflow_matmul(p_drop, vj)
                        else:
                            pv = torch.matmul(p_drop, vj)
                        o = o * upd_old + pv
                        st = st * upd_old + s_blk * upd_cur
                        mx = m_run_new
                    # Keep native per-segment normalization: 0/0 must remain
                    # NaN, even when the FD weight for that segment is zero.
                    segment_stats.append((o, st, mx))
                    mg_new = torch.maximum(m_g, mx)
                    sc_g = torch.exp(m_g - mg_new)
                    sc_g = torch.where(m_g <= minValue, torch.zeros_like(sc_g), sc_g)
                    sc_s = torch.exp(mx - mg_new)
                    sc_s = torch.where(mx <= minValue, torch.zeros_like(sc_s), sc_s)
                    o_g = o_g * sc_g + o * sc_s
                    s_g = s_g * sc_g + st * sc_s
                    m_g = mg_new

                max_all = torch.stack([item[2] for item in segment_stats]).amax(0)
                weights = [st * torch.exp(mx - max_all) for _, st, mx in segment_stats]
                total = torch.stack(weights).sum(0)
                if len(segment_stats) == 1:
                    res_rows = segment_stats[0][0] / segment_stats[0][1]
                else:
                    res_rows = torch.stack(
                        [
                            (o / st) * (weight / total)
                            for (o, st, _), weight in zip(segment_stats, weights)
                        ]
                    ).sum(0)
                # Match kernel InvalidRows for RIGHT_DOWN_CAUSAL. Rows
                # before q_len - kv_len have no valid keys, including those
                # inside a scheduled tile. Do not clear valid overflow rows.
                if causal_bound is not None:
                    res_rows = res_rows.masked_fill(causal_bound < 0, 0.0)
                out[bi, head_idx, s1_idx, :] = res_rows
                no_valid_score = m_valid_g <= mask_fill
                lse_rows = torch.where(
                    (m_g <= minValue).squeeze(-1) | no_valid_score,
                    torch.full_like(m_valid_g, float("inf")),
                    (m_g + torch.log(s_g)).squeeze(-1),
                )
                lse[bi, head_idx, s1_idx, 0] = lse_rows
    return out.contiguous(), lse.contiguous()


def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
):
    """CPU Flash Attention golden with MXFP8, C1V1C1V1C2V2 流水"""
    EPSILON = 1e-20
    Q_BLOCK_SIZE = 128
    if D == 256:
        K_BLOCK_SIZE = 128
        V_BLOCK_SIZE = 256
    else:
        K_BLOCK_SIZE = 256
        V_BLOCK_SIZE = 512

    # FP8 → FP32 反量化
    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q_tensor.shape[-1])

    # 尝试按 kernel SectionStreamK 分核调度模拟 (对齐 P 的 fp8 量化网格);
    # metadata 解码失败或计算异常时回退全局 online softmax 语义。
    # 注意: 必须在下方 GQA 广播之前分发 —— sectioned 按 kv head 索引
    # dequant_scale_k/v, 广播后的 scale 按 q head 排布, decode GQA 下
    # kv head 1..N_kv-1 会错拿 kv head 0 的 scale。
    section_info = get_qfa_section_info(actual_seq_q, actual_seq_kv)
    if section_info is not None:
        try:
            return _cpu_mxfp8_golden_sectioned(
                q_fp8,
                k_fp8,
                v_fp8,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                actual_seq_q,
                actual_seq_kv,
                section_info,
                softmax_scale=softmax_scale,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[Section] 分段 golden 计算失败(%s), 回退全局量化语义", exc)

    # GQA: 广播 K/V 到与 Q 相同的 head 数
    if N_q != N_kv:
        logger.info("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        dequant_scale_k = broadcast_kv(N_q, N_kv, dequant_scale_k)
        dequant_scale_v = broadcast_kv(N_q, N_kv, dequant_scale_v)

    b, n, s, d = q_tensor.shape

    dv = v_tensor.shape[-1]
    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]

    minValue = torch.tensor(-3.402823466e38, dtype=torch.float32)
    out = torch.zeros([b, n, Sq, dv], dtype=torch.float32)
    o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[
        ..., None
    ]

    TILES_Q = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    mask_global = _build_attention_mask(
        b, Sq, Skv, actual_seq_q, actual_seq_kv, SPARSE_MODE
    )

    Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
    V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
    o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    # dequant_scale 按 group_size 扩展，用于逐元素反量化
    dequant_scale_q_expanded = dequant_scale_q.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_k_expanded = dequant_scale_k.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_v_expanded = dequant_scale_v.repeat_interleave(
        QUANT_GROUP_SIZE, dim=2
    )[:, :, :, :D]

    logger.info(
        "[CPU Golden] TILES_Q=%d, TILES_KV=%d, Sq=%d, Skv=%d",
        TILES_Q,
        TILES_KV,
        Sq,
        Skv,
    )

    for i in range(TILES_Q):
        Qi = Q_BLOCKS[i]
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
        deq_scale_q_i = dequant_scale_q_expanded[:, :, Sq_start:Sq_end, :]

        for j in range(0, TILES_KV, 2):
            # C1V1C1V1C2V2: 每次迭代处理 2 个 K block + 1 个 V block
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            deq_scale_k_j = dequant_scale_k_expanded[:, :, Sk_start:Sk_end, :]

            S_ij = _compute_s_block(Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale)
            mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            m_block_j, s_block_j, P_ij_drop = _online_softmax_update(
                S_ij, mask_j, mi, si, oi, ln_p_scale
            )

            if j + 1 < TILES_KV:
                # --- 第二个 K block (j+1) ---
                Kj1 = K_BLOCKS[j + 1]
                Sk1_start = (j + 1) * K_BLOCK_SIZE
                Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                deq_scale_k_j1 = dequant_scale_k_expanded[:, :, Sk1_start:Sk1_end, :]

                S_ij1 = _compute_s_block(
                    Qi, Kj1, deq_scale_q_i, deq_scale_k_j1, softmax_scale
                )
                mask_j1 = mask_global[:, :, Sq_start:Sq_end, Sk1_start:Sk1_end]
                m_block_j1, s_block_j1, P_ij1_drop = _online_softmax_update(
                    S_ij1, mask_j1, m_block_j, s_block_j, oi, ln_p_scale
                )

                # V block: 一个 V_BLOCK_SIZE 对应两个 K_BLOCK_SIZE
                Vj = V_BLOCKS[j // 2]
                Sv_start = (j // 2) * V_BLOCK_SIZE
                Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                deq_scale_v_j = dequant_scale_v_expanded[:, :, Sv_start:Sv_end, :]
                Vj_dequant = Vj * deq_scale_v_j[:, :, : Vj.shape[2], :]

                V_part1 = Vj_dequant[:, :, : Kj.shape[2], :]
                V_part2 = Vj_dequant[:, :, Kj.shape[2] : Kj.shape[2] + Kj1.shape[2], :]
                P_ij_Vj = torch.matmul(
                    P_ij_drop * torch.exp(m_block_j - m_block_j1), V_part1
                ) + torch.matmul(P_ij1_drop, V_part2)

                update_mul_si = torch.exp(mi - m_block_j1)
                si_new = (
                    update_mul_si * si
                    + s_block_j * torch.exp(m_block_j - m_block_j1)
                    + s_block_j1
                )
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j1
            else:
                Vj = V_BLOCKS[j // 2]
                Sv_start = j * K_BLOCK_SIZE
                Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)
                deq_scale_v_j = dequant_scale_v_expanded[:, :, Sv_start:Sv_end, :]
                Vj_expanded = Vj[:, :, : Kj.shape[2], :]
                Vj_dequant = Vj_expanded * deq_scale_v_j[:, :, : Kj.shape[2], :]

                P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant)
                update_mul_si = torch.exp(mi - m_block_j)
                si_new = update_mul_si * si + s_block_j
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j

    out = torch.cat(o_BLOCKS, dim=2)
    out_sum = torch.cat(s_BLOCKS, dim=2)
    out = out / (out_sum + EPSILON)

    o_max = torch.cat(m_BLOCKS, dim=2)
    all_masked = o_max <= minValue.item()
    lse = torch.where(
        all_masked,
        torch.full_like(o_max, float("inf")),
        o_max + torch.log(out_sum + EPSILON),
    )
    out = torch.where(all_masked, torch.zeros_like(out), out)
    logger.info("[CPU Golden] output=%s", out.shape)
    return out, lse


# ==============================================================================
# NPU 调用
# GRAPH_PATH: 0=单算子, 3=静态图, 5=动态图, 6=tiling下沉, 7=aclgraph
# PA 模式: Q 用 TND layout, K/V 走 PA 预处理 (block_table + block_size)
# 非 PA 模式: Q/K/V 均按 INPUT_LAYOUT 转换
# ==============================================================================


def _build_causal_mask():
    # sparse_mode=0 不需要 mask，其他模式需要上三角 causal mask
    if SPARSE_MODE == 0:
        return None
    shape = ATTN_MASK_SHAPE
    if not shape:
        shape = (2048, 2048)
    return torch.triu(torch.ones(shape, dtype=torch.int8), diagonal=1).npu()


def prepare_npu_inputs(
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
    """准备 NPU 侧入参

    返回字典的 key 与 _call_npu_fa_op 的形参名一一对应:
      q, k, v, mask,
      cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv,
      dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
      block_table, q_n, kv_n, softmax_scale,
      layout_q, layout_q_descale, layout_kv, layout_out, block_size, sparse_mode, out_dtype
    其中 cu_seqlens_q/kv、seqused_q/kv 为 python list (或 None), 由 _call_npu_fa_op 负责转 NPU tensor;
    其余 tensor 字段均为已就绪的 NPU tensor.
    """
    torch_npu.npu.set_device(int(DEVICE_ID))

    softmax_scale = SOFTMAX_SCALE

    Q_DTYPE = q_fp8.dtype
    DQ_DTYPE = dequant_scale_q.dtype
    P_SCALE_DTYPE = p_scale.dtype
    q_npu = q_fp8.contiguous().view(Q_DTYPE).npu()
    deq_q_npu = dequant_scale_q.view(DQ_DTYPE).npu()
    p_scale_npu = p_scale.view(P_SCALE_DTYPE).npu()

    out_dtype = torch.float16
    mask_arg = _build_causal_mask()

    if ENABLE_PA:
        K_DTYPE = k_fp8.dtype
        V_DTYPE = v_fp8.dtype
        DK_DTYPE = dequant_scale_k.dtype
        DV_DTYPE = dequant_scale_v.dtype
        k_npu = k_fp8.view(K_DTYPE).npu()
        v_npu = v_fp8.view(V_DTYPE).npu()
        deq_k_npu = dequant_scale_k.view(DK_DTYPE).npu()
        deq_v_npu = dequant_scale_v.view(DV_DTYPE).npu()

        logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
        logger.info("[NPU PA] k=%s, v=%s", k_npu.shape, v_npu.shape)
        logger.info("[NPU PA] k stride=%s, v stride=%s", k_npu.stride(), v_npu.stride())
        logger.info("[NPU PA] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)
        logger.info(
            "[NPU PA] deq_k stride=%s, deq_v stride=%s",
            deq_k_npu.stride(),
            deq_v_npu.stride(),
        )

        block_table_npu = (
            block_table_torch.npu()
            if isinstance(block_table_torch, torch.Tensor)
            else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
        )

        _pa_layout_kv_map = {"BnNBsD": "PA_BNBD", "BnBsND": "PA_BBND", "PA_NZ": "PA_NZ"}
        pa_layout_kv = _pa_layout_kv_map.get(KV_CACHE_LAYOUT, "PA_BNBD")

        # layout_kv 优先用 CSV 透传值 (LAYOUT_KV); 若未注入则回退到 kv_cache_layout 推导
        op_layout_kv = LAYOUT_KV if LAYOUT_KV else pa_layout_kv

        logger.info("[NPU] prepare PA inputs done.")
        return dict(
            q=q_npu,
            k=k_npu,
            v=v_npu,
            mask=mask_arg,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            cu_seqlens_q_t=cu_seqlens_q_t,
            cu_seqlens_kv_t=cu_seqlens_kv_t,
            seqused_q_t=seqused_q_t,
            seqused_kv_t=seqused_kv_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            dequant_scale_q=deq_q_npu,
            dequant_scale_k=deq_k_npu,
            dequant_scale_v=deq_v_npu,
            p_scale=p_scale_npu,
            block_table=block_table_npu,
            q_n=N_q,
            kv_n=N_kv,
            softmax_scale=softmax_scale,
            layout_q=LAYOUT_Q,
            layout_q_descale=LAYOUT_Q_DESCALE,
            layout_kv=op_layout_kv,
            layout_out=LAYOUT_OUT,
            block_size=BLOCK_SIZE,
            sparse_mode=SPARSE_MODE,
            out_dtype=out_dtype,
        )

    # 非 PA 模式
    K_DTYPE = k_fp8.dtype
    V_DTYPE = v_fp8.dtype
    DK_DTYPE = dequant_scale_k.dtype
    DV_DTYPE = dequant_scale_v.dtype
    k_npu = k_fp8.view(K_DTYPE).npu()
    v_npu = v_fp8.view(V_DTYPE).npu()
    deq_k_npu = dequant_scale_k.view(DK_DTYPE).npu()
    deq_v_npu = dequant_scale_v.view(DV_DTYPE).npu()
    logger.info("[NPU TND] k=%s, v=%s", k_npu.shape, v_npu.shape)
    logger.info("[NPU TND] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)
    logger.info(
        "[NPU TND] k is_contiguous=%s, v is_contiguous=%s",
        k_npu.is_contiguous(),
        v_npu.is_contiguous(),
    )
    logger.info(
        "[NPU TND] deq_k is_contiguous=%s, deq_v is_contiguous=%s",
        deq_k_npu.is_contiguous(),
        deq_v_npu.is_contiguous(),
    )

    logger.info("[NPU] prepare non-PA inputs done.")
    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        cu_seqlens_q_t=cu_seqlens_q_t,
        cu_seqlens_kv_t=cu_seqlens_kv_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=deq_k_npu,
        dequant_scale_v=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=None,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q=LAYOUT_Q,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=LAYOUT_KV,
        layout_out=LAYOUT_OUT,
        block_size=0,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
    )
