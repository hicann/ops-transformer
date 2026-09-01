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
支持：PA / 非PA 场景，rope 分离传入，GQA
量化：Q/K per-token-group (quant_mode=6), V per-channel-group (quant_mode=8)
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import argparse
import logging
import math

import torch
import torch.nn as nn
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    _HAS_NPU = True
except ImportError as e:
    logger.warning("Failed to import cann_ops_transformer.ops: %s", e)
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

# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 7=aclgraph
# 优先从环境变量读取，支持 GRAPH_PATH=7 python3 -m pytest ...
import os as _os

GRAPH_PATH = int(_os.environ.get("GRAPH_PATH", "0"))
B = 1
N_q = 1  # query heads
N_kv = 1  # kv heads
D = 128

ENABLE_ROPE = False
D_rope = 64

CU_SEQLENS_Q = [0, 4]
CU_SEQLENS_KV = [0, 1024]
SEQUSED_Q = [4]
SEQUSED_KV = [1024]
MAX_SEQLEN_Q = 4
MAX_SEQLEN_KV = 1024


def _derive_seqused(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _get_seqused_q():
    return SEQUSED_Q if SEQUSED_Q is not None else _derive_seqused(CU_SEQLENS_Q)


def _get_seqused_kv():
    return SEQUSED_KV if SEQUSED_KV is not None else _derive_seqused(CU_SEQLENS_KV)


ENABLE_PA = False
BLOCK_SIZE = 512

SPARSE_MODE = 3
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 32

# Layout 选择
INPUT_LAYOUT = "TND"  # 支持: BNSD, BSND, TND
Q_SCALE_LAYOUT = "TND"  # 支持: TND, N2TGD
P_SCALE = 1.0

SOFTMAX_SCALE = None

# PA KV Cache Layout
# BnNBsD: [BlockNum, N, BlockSize, D]
# PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
KV_CACHE_LAYOUT = "BnNBsD"

IS_CONTIGUOUS = True

ENABLE_LSE = False

# e8m0fnu 最小正数: 2^(-127)，用于替换 scale 中的 0 和非有限值
# e8m0fnu 没有 0 值语义，0 的 biased exponent 会被 NPU 解释为 NaN
E8M0_MIN_POSITIVE = 2 ** (-127)

SEED_Q = 54
SEED_K = 3
SEED_V = 4
SEED_QR = 8
SEED_KR = 9

DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0
DATA_RANGE_QR = 1.0
DATA_RANGE_KR = 1.0

DEVICE_ID = 0


def _get_npu_fa_kwargs():
    # 每次调用时读取最新全局变量，确保 pytest 修改配置后生效
    return dict(
        return_softmax_lse=ENABLE_LSE,
    )


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


def convert_qk_rope_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    return convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout)


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


def canonical_q_scale_layout(layout):
    layout = (layout or "TND").upper()
    if layout not in ("TND", "N2TGD"):
        raise ValueError(f"Unsupported Q scale layout: {layout}")
    return layout


def resolve_q_scale_layout(layout=None):
    resolved = canonical_q_scale_layout(layout or Q_SCALE_LAYOUT)
    if N_kv <= 0 or N_q % N_kv != 0:
        raise ValueError(f"N_q must be divisible by N_kv, got N_q={N_q}, N_kv={N_kv}")
    group = N_q // N_kv
    return resolved, group


def pack_qk_scale_for_npu(scale_flat):
    """Q/K scale packing: (..., D) → (..., D//2, 2)
    NPU 要求 Q/K scale 按 (偶, 奇) 对打包，每两个相邻 scale 值合并为一个 [..., 2]
    """
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def pack_v_scale_for_npu(scale_flat):
    """V scale packing: (..., Sg, D) → (..., Sg//2, D, 2)
    NPU 要求 V scale 按行 (偶行, 奇行) 交错打包
    奇数行 pad 用 E8M0_MIN_POSITIVE，避免 0.0 转 e8m0fnu 后变 NaN
    """
    Sg = scale_flat.shape[-2]
    D = scale_flat.shape[-1]
    prefix_shape = scale_flat.shape[:-2]

    # 奇数行 pad 用 E8M0_MIN_POSITIVE，避免 0.0 转 e8m0fnu 后变 NaN
    if Sg % 2 != 0:
        pad_shape = prefix_shape + (1, D)
        pad = torch.full(
            pad_shape,
            E8M0_MIN_POSITIVE,
            dtype=scale_flat.dtype,
            device=scale_flat.device,
        )
        scale_flat = torch.cat([scale_flat, pad], dim=-2)
        Sg += 1

    out_Sg = Sg // 2
    out_shape = prefix_shape + (out_Sg, D, 2)

    result = torch.zeros(out_shape, dtype=scale_flat.dtype, device=scale_flat.device)
    result[..., 0] = scale_flat[..., ::2, :]
    result[..., 1] = scale_flat[..., 1::2, :]
    return result


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


def convert_q_scale_tnd_to_n2gtd_layout(tensor_tnd, num_kv_heads):
    return convert_q_scale_tnd_to_n2tgd_layout(tensor_tnd, num_kv_heads)


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
    is_rope=False,
    kv_layout="BnNBsD",
    group_size=32,
):
    """
    MXFP8 PA 预处理: BNSD → PagedAttention KV Cache

    输入: [B, N, S, D]
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K scale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V scale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], Kscale=[Bn,N,Bs,D//64,2], Vscale=[Bn,N,Bs//64,D,2]
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

    if is_rope and is_scale:
        raise ValueError("rope preprocessing does not support scale packing")

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_scale_pack_ratio = group_size * 2
            pack_seq_lens = [
                math.ceil(act_s / v_scale_pack_ratio) for act_s in seq_lens
            ]
            pack_block_size = math.ceil(block_size / v_scale_pack_ratio)
            total_block_num = sum(
                math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens
            )
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K scale D must be even for packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (
                sum(math.ceil(act_s / block_size) for act_s in seq_lens),
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
            sum(math.ceil(act_s / block_size) for act_s in seq_lens),
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

    if kv_layout == "BnNBsD":
        return out_cache
    elif kv_layout == "BnBsND":
        return out_cache.transpose(1, 2).contiguous()
    elif kv_layout == "PA_NZ":
        Bn, KV_N, Bs, KV_D = out_cache.shape[:4]
        if not is_scale:
            inner = 16 if is_rope else 32
            if KV_D % inner != 0:
                raise ValueError(
                    f"PA_NZ rope D must be divisible by {inner}, got {KV_D}"
                )
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


def bnsd_to_pa_kv(tensor_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD"):
    return mxfp8_pa_preprocessing(
        tensor_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=False,
        kv_layout=kv_layout,
    )


def bnsd_to_pa_kv_scale_token_group(
    scale_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD", group_size=32
):
    return mxfp8_pa_preprocessing(
        scale_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=True,
        is_vscale=False,
        kv_layout=kv_layout,
        group_size=group_size,
    )


def bnsd_to_pa_v_scale_channel_group(
    scale_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD", group_size=32
):
    return mxfp8_pa_preprocessing(
        scale_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=True,
        is_vscale=True,
        kv_layout=kv_layout,
        group_size=group_size,
    )


def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


# ==============================================================================
# 数据生成
# ==============================================================================


def generate_data():
    """生成 BNSD FP16 Q/K/V 并做 MXFP8 量化，使用原始序列长度"""
    max_sq = MAX_SEQLEN_Q
    max_skv = MAX_SEQLEN_KV
    seqused_q = _get_seqused_q()
    seqused_kv = _get_seqused_kv()
    if max_sq is None or max_sq < 0:
        max_sq = max(seqused_q)
    if max_skv is None or max_skv < 0:
        max_skv = max(seqused_kv)

    logger.info("[INFO] max_sq=%d, max_skv=%d", max_sq, max_skv)

    torch.manual_seed(SEED_Q)
    q_fp16 = (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_Q

    torch.manual_seed(SEED_K)
    k_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * DATA_RANGE_K

    torch.manual_seed(SEED_V)
    v_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * DATA_RANGE_V

    qr_bf16 = None
    kr_bf16 = None
    if ENABLE_ROPE and D_rope > 0:
        torch.manual_seed(SEED_QR)
        qr_bf16 = (
            torch.rand(B, N_q, max_sq, D_rope, dtype=torch.bfloat16) * 2 - 1
        ) * DATA_RANGE_QR
        torch.manual_seed(SEED_KR)
        kr_bf16 = (
            torch.rand(B, N_kv, max_skv, D_rope, dtype=torch.bfloat16) * 2 - 1
        ) * DATA_RANGE_KR

    logger.info(
        "[INFO] q_fp16=%s, k_fp16=%s, v_fp16=%s",
        q_fp16.shape,
        k_fp16.shape,
        v_fp16.shape,
    )

    quant_scale_q = get_mxfp8_per_token_group_quant_scale(
        q_fp16, FP8_DTYPE, QUANT_GROUP_SIZE
    )
    quant_scale_k = get_mxfp8_per_token_group_quant_scale(
        k_fp16, FP8_DTYPE, QUANT_GROUP_SIZE
    )
    quant_scale_v = get_mxfp8_per_channel_group_quant_scale(
        v_fp16, FP8_DTYPE, QUANT_GROUP_SIZE
    )

    dequant_scale_q = quant_scale_q
    dequant_scale_k = quant_scale_k
    v_descale = quant_scale_v

    v_sg = v_descale.shape[2]
    logger.info("[INFO] V scale Sg=%d, 是否奇数=%s", v_sg, v_sg % 2 != 0)

    fp8_max = 448.0 if FP8_DTYPE == torch.float8_e4m3fn else 57344.0
    q_fp8 = (
        mxfp8_per_token_group_quant(q_fp16, quant_scale_q, QUANT_GROUP_SIZE)
        .clamp(-fp8_max, fp8_max)
        .to(FP8_DTYPE)
    )
    k_fp8 = (
        mxfp8_per_token_group_quant(k_fp16, quant_scale_k, QUANT_GROUP_SIZE)
        .clamp(-fp8_max, fp8_max)
        .to(FP8_DTYPE)
    )
    v_fp8 = (
        mxfp8_per_channel_group_quant(v_fp16, quant_scale_v, QUANT_GROUP_SIZE)
        .clamp(-fp8_max, fp8_max)
        .to(FP8_DTYPE)
    )

    block_table_torch = None
    if ENABLE_PA:
        block_num = sum(math.ceil(s / BLOCK_SIZE) for s in seqused_kv)
        max_blocks = max(math.ceil(s / BLOCK_SIZE) for s in seqused_kv)
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        block_table_torch = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(seqused_kv[b] / BLOCK_SIZE)
            for j in range(n_blocks):
                block_table_torch[b, j] = block_idx_list[idx]
                idx += 1

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    return (
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        qr_bf16,
        kr_bf16,
        block_table_torch,
    )


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


def _compute_s_block(
    Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale, Qri=None, Krj=None
):
    """计算单个 S block (attention score)"""
    S_ij = torch.matmul(Qi * deq_scale_q_i, (Kj * deq_scale_k_j).permute(0, 1, 3, 2))
    if Qri is not None and Krj is not None:
        S_ij += torch.matmul(
            Qri.to(torch.float32), Krj.to(torch.float32).permute(0, 1, 3, 2)
        )
    return S_ij * softmax_scale


def _online_softmax_update(S_ij, mask_j, mi, si, oi, ln_p_scale, LN2):
    """Online softmax: 计算 m, P, s 更新 (MXFP8: stored max 不含 -ln(p_scale), P 含 p_scale 因子)
    1. mask 位置填 -inf
    2. 求 block 内 max (m_block_j)
    3. m 对齐到 ln2 整数倍 (ceil)，模拟 NPU MXFP8 量化精度损失
    4. 与前一个 block 的 m 取 max (stored max 不含 -ln(p_scale))
    5. 计算 P = exp(S - m + ln_p_scale)，模拟 NPU: exp 用 adjusted max (含 -ln(p_scale))，但 stored max 不含
    6. 求 s = sum(P)
    7. P 转 FP8 再转回 FP32，模拟 NPU 侧 P 的量化损失
    """
    S_ij = S_ij.masked_fill(mask_j, float("-inf"))

    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
    m_block_j = torch.ceil(m_block_j / LN2) * LN2
    m_block_j = torch.max(mi, m_block_j)

    P_ij_raw = torch.exp(S_ij - m_block_j + ln_p_scale)
    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
    P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)

    return m_block_j, s_block_j, P_ij_drop


def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """CPU Flash Attention golden with MXFP8, C1V1C1V1C2V2 流水"""
    EPSILON = 1e-20
    LN2 = math.log(2.0)
    Q_BLOCK_SIZE = 128
    if D == 256:
        K_BLOCK_SIZE = 128
        V_BLOCK_SIZE = 256
    else:
        K_BLOCK_SIZE = 256
        V_BLOCK_SIZE = 512

    if actual_seq_q is None:
        actual_seq_q = _get_seqused_q()
    if actual_seq_kv is None:
        actual_seq_kv = _get_seqused_kv()

    # FP8 → FP32 反量化
    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    # GQA: 广播 K/V 到与 Q 相同的 head 数
    if N_q != N_kv:
        logger.info("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        dequant_scale_k = broadcast_kv(N_q, N_kv, dequant_scale_k)
        v_descale = broadcast_kv(N_q, N_kv, v_descale)
        if kr_bf16 is not None:
            kr_bf16 = broadcast_kv(N_q, N_kv, kr_bf16)

    b, n, s, d = q_tensor.shape
    d_total = d + (D_rope if ENABLE_ROPE else 0)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_total)
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

    Qr_BLOCKS = None
    Kr_BLOCKS = None
    if ENABLE_ROPE and qr_bf16 is not None:
        Qr_BLOCKS = list(torch.split(qr_bf16, Q_BLOCK_SIZE, dim=2))
        Kr_BLOCKS = list(torch.split(kr_bf16, K_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    # dequant_scale 按 group_size 扩展，用于逐元素反量化
    dequant_scale_q_expanded = dequant_scale_q.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_k_expanded = dequant_scale_k.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    v_descale_expanded = v_descale.repeat_interleave(QUANT_GROUP_SIZE, dim=2)

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

        Qri = Qr_BLOCKS[i] if Qr_BLOCKS is not None else None

        for j in range(0, TILES_KV, 2):
            # C1V1C1V1C2V2: 每次迭代处理 2 个 K block + 1 个 V block
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            deq_scale_k_j = dequant_scale_k_expanded[:, :, Sk_start:Sk_end, :]
            Krj = Kr_BLOCKS[j] if Kr_BLOCKS is not None else None

            S_ij = _compute_s_block(
                Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale, Qri, Krj
            )
            mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            m_block_j, s_block_j, P_ij_drop = _online_softmax_update(
                S_ij, mask_j, mi, si, oi, ln_p_scale, LN2
            )

            if j + 1 < TILES_KV:
                # --- 第二个 K block (j+1) ---
                Kj1 = K_BLOCKS[j + 1]
                Sk1_start = (j + 1) * K_BLOCK_SIZE
                Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                deq_scale_k_j1 = dequant_scale_k_expanded[:, :, Sk1_start:Sk1_end, :]
                Krj1 = Kr_BLOCKS[j + 1] if Kr_BLOCKS is not None else None

                S_ij1 = _compute_s_block(
                    Qi, Kj1, deq_scale_q_i, deq_scale_k_j1, softmax_scale, Qri, Krj1
                )
                mask_j1 = mask_global[:, :, Sq_start:Sq_end, Sk1_start:Sk1_end]
                m_block_j1, s_block_j1, P_ij1_drop = _online_softmax_update(
                    S_ij1, mask_j1, m_block_j, s_block_j, oi, ln_p_scale, LN2
                )

                # V block: 一个 V_BLOCK_SIZE 对应两个 K_BLOCK_SIZE
                Vj = V_BLOCKS[j // 2]
                Sv_start = (j // 2) * V_BLOCK_SIZE
                Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                deq_scale_v_j = v_descale_expanded[:, :, Sv_start:Sv_end, :]
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
                deq_scale_v_j = v_descale_expanded[:, :, Sv_start:Sv_end, :]
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


def _call_npu_fa_op(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
):
    """调用 NPU 双算子 (QFA: quant_flash_attn_metadata + quant_flash_attn)"""
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
    cu_seqlens_kv_t = (
        torch.tensor(cu_seqlens_kv, dtype=torch.int32).npu()
        if cu_seqlens_kv is not None
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

    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=q_n,
        num_heads_kv=kv_n,
        head_dim=q.shape[-1],
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        mask_mode=sparse_mode,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )

    main_kwargs = dict(
        q=q,
        k=k,
        v=v,
        q_descale=dequant_scale_q,
        k_descale=dequant_scale_k,
        v_descale=v_descale,
        quant_mode=1,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=sparse_mode,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    main_kwargs.update(_get_npu_fa_kwargs())
    atten_out, lse_out = quant_flash_attn(**main_kwargs)
    torch.npu.synchronize()
    return atten_out, lse_out


class Network(nn.Module):
    """aclgraph 编译目标: forward 只包含两个 torch.library op 调用。

    输入已预处理:
      - cu_seqlens_q/kv, seqused_q/kv: 由 mxfp8_fa_torch_npu 转好 NPU tensor
      - q/k/v/deq_*: 已转好 layout 的 NPU tensor
      - max_seqlen_q/kv: 已计算好的标量
    """

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
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        block_size,
        sparse_mode,
        out_dtype,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        metadata = quant_flash_attn_metadata(
            num_heads_q=q_n,
            num_heads_kv=kv_n,
            head_dim=q.shape[-1],
            quant_mode=1,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            head_dim_v=None,
            mask_mode=sparse_mode,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        main_kwargs = dict(
            q=q,
            k=k,
            v=v,
            q_descale=dequant_scale_q,
            k_descale=dequant_scale_k,
            v_descale=v_descale,
            quant_mode=1,
            block_table=block_table,
            p_scale=p_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            mask_mode=sparse_mode,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        main_kwargs.update(_get_npu_fa_kwargs())
        atten_out, lse_out = quant_flash_attn(**main_kwargs)
        return atten_out, lse_out


def _build_causal_mask():
    # sparse_mode=0 不需要 mask，其他模式需要上三角 causal mask
    if SPARSE_MODE == 0:
        return None
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def _prepare_rope_npu(
    qr_bf16,
    kr_bf16,
    actual_seq_q,
    actual_seq_kv,
    npu_input_layout,
    block_table_torch,
    enable_pa,
):
    """准备 rope 数据的 NPU tensor"""
    if not ENABLE_ROPE or qr_bf16 is None or kr_bf16 is None:
        if ENABLE_ROPE:
            raise ValueError("ENABLE_ROPE=True requires qr_bf16 and kr_bf16")
        return None, None

    q_rope_npu = convert_qk_rope_bnsd_to_layout(
        qr_bf16, actual_seq_q, npu_input_layout
    ).npu()

    if enable_pa:
        k_rope_pa = mxfp8_pa_preprocessing(
            kr_bf16,
            actual_seq_kv,
            BLOCK_SIZE,
            block_table_torch,
            is_rope=True,
            kv_layout=KV_CACHE_LAYOUT,
        )
        k_rope_npu = k_rope_pa.npu()
        if not IS_CONTIGUOUS:
            # ---- 构造krope不连续 ----
            fake_krope_tensor = torch.ones_like(k_rope_pa)
            double_krope = torch.stack([k_rope_pa, fake_krope_tensor], dim=2)
            double_krope = double_krope.npu()
            k_rope_npu = double_krope[:, :, 0]  # 覆写为非连续
            logger.info(f"[NPU] k_rope is_contiguous={k_rope_npu.is_contiguous()}")
    else:
        k_rope_npu = convert_qk_rope_bnsd_to_layout(
            kr_bf16, actual_seq_kv, npu_input_layout
        ).npu()

    logger.info("[NPU] q_rope=%s, k_rope=%s", q_rope_npu.shape, k_rope_npu.shape)
    return q_rope_npu, k_rope_npu


def prepare_npu_inputs(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """准备 NPU 侧入参 (layout 转换 / scale 转换 / mask / rope / block_table)，返回可直接传给
    _call_npu_fa_op 的入参字典。

    返回字典的 key 与 _call_npu_fa_op 的形参名一一对应：
      q, k, v, q_rope, k_rope, mask,
      cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv,
      dequant_scale_q, dequant_scale_k, v_descale, p_scale,
      block_table, q_n, kv_n, softmax_scale,
      layout_q, layout_q_descale, layout_kv, layout_out, block_size, sparse_mode, out_dtype
    其中 cu_seqlens_q/kv、seqused_q/kv 为 python list (或 None)，由 _call_npu_fa_op 负责转 NPU tensor；
    其余 tensor 字段均为已就绪的 NPU tensor。
    """
    torch_npu.npu.set_device(int(DEVICE_ID))

    softmax_scale = (
        SOFTMAX_SCALE
        if SOFTMAX_SCALE is not None
        else 1.0 / math.sqrt(D + (D_rope if ENABLE_ROPE else 0))
    )

    q_runtime_layout, q_group = resolve_q_scale_layout()
    logger.info("[NPU] Q layout: q_layout=%s, G=%d", q_runtime_layout, q_group)

    npu_input_layout = "TND" if ENABLE_PA else INPUT_LAYOUT
    act_seqused_q = _get_seqused_q()
    act_seqused_kv = _get_seqused_kv()
    q_npu = (
        convert_q_bnsd_to_layout(
            q_fp8, act_seqused_q, npu_input_layout, cu_seqlens=CU_SEQLENS_Q
        )
        .contiguous()
        .view(FP8_DTYPE)
        .npu()
    )
    logger.info("[NPU %s] q=%s", npu_input_layout, q_npu.shape)

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(
        convert_q_scale_bnsd_to_layout(
            dequant_scale_q, act_seqused_q, q_runtime_layout, cu_seqlens=CU_SEQLENS_Q
        ),
        "Q scale",
    )
    deq_q_npu = q_scale_e8m0.npu()
    logger.info(
        "[NPU] Q scale layout=%s, shape=%s", q_runtime_layout, q_scale_e8m0.shape
    )

    p_scale_npu = p_scale.npu()

    out_dtype = torch.bfloat16 if ENABLE_ROPE else torch.float16

    mask_arg = _build_causal_mask()
    q_rope_npu, k_rope_npu = _prepare_rope_npu(
        qr_bf16,
        kr_bf16,
        act_seqused_q,
        act_seqused_kv,
        npu_input_layout,
        block_table_torch,
        ENABLE_PA,
    )

    if ENABLE_PA:
        # PA 模式: K/V 走 mxfp8_pa_preprocessing 转换为 PagedAttention KV Cache 格式
        k_pa = mxfp8_pa_preprocessing(
            k_fp8,
            act_seqused_kv,
            BLOCK_SIZE,
            block_table_torch,
            is_scale=False,
            kv_layout=KV_CACHE_LAYOUT,
        )
        v_pa = (
            mxfp8_pa_preprocessing(
                v_fp8,
                act_seqused_kv,
                BLOCK_SIZE,
                block_table_torch,
                is_scale=False,
                kv_layout=KV_CACHE_LAYOUT,
            )
            if v_fp8 is not None
            else None
        )
        k_npu = k_pa.contiguous().view(FP8_DTYPE).npu()
        v_npu = v_pa.contiguous().view(FP8_DTYPE).npu() if v_pa is not None else None
        if not IS_CONTIGUOUS:
            # ---- 构造kv不连续 ----
            if v_pa is not None:
                kv_cache = torch.stack([k_pa, v_pa], dim=2)
                kv_cache = kv_cache.npu()
                k_npu = kv_cache[:, :, 0]
                v_npu = kv_cache[:, :, 1]
            logger.info(
                f"[NPU] key is_contiguous={k_npu.is_contiguous()}, value is_contiguous={v_npu.is_contiguous() if v_npu is not None else 'N/A'}"
            )

        k_scale_pa = mxfp8_pa_preprocessing(
            dequant_scale_k,
            act_seqused_kv,
            BLOCK_SIZE,
            block_table_torch,
            is_scale=True,
            is_vscale=False,
            kv_layout=KV_CACHE_LAYOUT,
        )
        v_scale_pa = (
            mxfp8_pa_preprocessing(
                v_descale,
                act_seqused_kv,
                BLOCK_SIZE,
                block_table_torch,
                is_scale=True,
                is_vscale=True,
                kv_layout=KV_CACHE_LAYOUT,
            )
            if v_descale is not None
            else None
        )

        k_scale_e8m0_pa = fp32_to_e8m0fnu_safe(k_scale_pa, "K PA scale")
        v_scale_e8m0_pa = (
            fp32_to_e8m0fnu_safe(v_scale_pa, "V PA scale")
            if v_scale_pa is not None
            else None
        )

        deq_k_npu = k_scale_e8m0_pa.npu()
        deq_v_npu = v_scale_e8m0_pa.npu() if v_scale_e8m0_pa is not None else None
        # ---- 构造kvscale不连续 ----
        if not IS_CONTIGUOUS:
            fake_kscale_tensor = torch.ones_like(k_scale_e8m0_pa)
            double_kscale = torch.stack([k_scale_e8m0_pa, fake_kscale_tensor], dim=2)
            double_kscale = double_kscale.npu()
            deq_k_npu = double_kscale[:, :, 0]  # 覆写为非连续
            if v_scale_e8m0_pa is not None:
                fake_vscale_tensor = torch.ones_like(v_scale_e8m0_pa)
                double_vscale = torch.stack(
                    [v_scale_e8m0_pa, fake_vscale_tensor], dim=2
                )
                double_vscale = double_vscale.npu()
                deq_v_npu = double_vscale[:, :, 0]
            logger.info(
                f"[NPU] deq_k_scale is_contiguous={deq_k_npu.is_contiguous()}, deq_v_scale is_contiguous={deq_v_npu.is_contiguous() if deq_v_npu is not None else 'N/A'}"
            )

        logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
        logger.info(
            "[NPU PA] k=%s, v=%s",
            k_npu.shape,
            v_npu.shape if v_npu is not None else None,
        )
        logger.info(
            "[NPU PA] deq_k=%s, deq_v=%s",
            deq_k_npu.shape,
            deq_v_npu.shape if deq_v_npu is not None else None,
        )

        block_table_npu = (
            block_table_torch.npu()
            if isinstance(block_table_torch, torch.Tensor)
            else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
        )

        _pa_layout_kv_map = {"BnNBsD": "PA_BNBD", "BnBsND": "PA_BBND", "PA_NZ": "PA_NZ"}
        pa_layout_kv = _pa_layout_kv_map.get(KV_CACHE_LAYOUT, "PA_BNBD")

        logger.info("[NPU] prepare PA inputs done.")
        return dict(
            q=q_npu,
            k=k_npu,
            v=v_npu,
            q_rope=q_rope_npu,
            k_rope=k_rope_npu,
            mask=mask_arg,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            dequant_scale_q=deq_q_npu,
            dequant_scale_k=deq_k_npu,
            v_descale=deq_v_npu,
            p_scale=p_scale_npu,
            block_table=block_table_npu,
            q_n=N_q,
            kv_n=N_kv,
            softmax_scale=softmax_scale,
            layout_q="TND",
            layout_q_descale=q_runtime_layout,
            layout_kv=pa_layout_kv,
            layout_out="TND",
            block_size=BLOCK_SIZE,
            sparse_mode=SPARSE_MODE,
            out_dtype=out_dtype,
        )

    # 非 PA 模式: K/V 按 INPUT_LAYOUT 转换
    k_npu = (
        convert_kv_bnsd_to_layout(
            k_fp8, act_seqused_kv, npu_input_layout, cu_seqlens=CU_SEQLENS_KV
        )
        .contiguous()
        .view(FP8_DTYPE)
        .npu()
    )
    v_npu = (
        convert_kv_bnsd_to_layout(
            v_fp8, act_seqused_kv, npu_input_layout, cu_seqlens=CU_SEQLENS_KV
        )
        .contiguous()
        .view(FP8_DTYPE)
        .npu()
        if v_fp8 is not None
        else None
    )
    logger.info(
        "[NPU %s] k=%s, v=%s",
        npu_input_layout,
        k_npu.shape,
        v_npu.shape if v_npu is not None else None,
    )

    k_scale_e8m0 = fp32_to_e8m0fnu_safe(
        convert_k_scale_bnsd_to_layout(
            dequant_scale_k, act_seqused_kv, npu_input_layout, cu_seqlens=CU_SEQLENS_KV
        ),
        "K scale",
    )
    v_scale_e8m0 = (
        fp32_to_e8m0fnu_safe(
            convert_v_scale_bnsd_to_layout(v_descale, act_seqused_kv, npu_input_layout),
            "V scale",
        )
        if v_descale is not None
        else None
    )
    deq_k_npu = k_scale_e8m0.npu()
    deq_v_npu = v_scale_e8m0.npu() if v_scale_e8m0 is not None else None
    logger.info(
        "[NPU %s] K scale shape=%s, V scale shape=%s",
        npu_input_layout,
        k_scale_e8m0.shape,
        v_scale_e8m0.shape if v_scale_e8m0 is not None else None,
    )

    logger.info("[NPU] prepare non-PA inputs done.")
    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        q_rope=q_rope_npu,
        k_rope=k_rope_npu,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=deq_k_npu,
        v_descale=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=None,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q=npu_input_layout,
        layout_q_descale=q_runtime_layout,
        layout_kv=npu_input_layout,
        layout_out=npu_input_layout,
        block_size=0,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
    )


def npu_mxfp8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """调用 NPU 算子，支持 N2TGD layout

    正常用例入口：准备入参 → 调用算子 → 输出截断/LSE 处理。
    """
    inputs = prepare_npu_inputs(
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        block_table_torch=block_table_torch,
        qr_bf16=qr_bf16,
        kr_bf16=kr_bf16,
    )

    logger.info(
        "[NPU] 调用 %s 模式 (GRAPH_PATH=%d)...",
        "PA" if ENABLE_PA else inputs["layout_q"],
        GRAPH_PATH,
    )
    # 通过 mxfp8_fa_torch_npu 路由，支持 GRAPH_PATH=7 (aclgraph)
    atten_out, lse_out = mxfp8_fa_torch_npu(**inputs)

    act_seqused_q = _get_seqused_q()
    npu_output = atten_out
    T_actual = cu_seqlens_q[-1] if cu_seqlens_q is not None else sum(act_seqused_q)
    if npu_output.shape[0] > T_actual:
        npu_output = npu_output[:T_actual]
    logger.info("[NPU] output=%s", npu_output.shape)
    return npu_output, lse_out


def mxfp8_fa_torch_npu(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
):
    """NPU 调用入口, 支持 GRAPH_PATH=0 (单算子) 和 GRAPH_PATH=7 (aclgraph)"""
    if GRAPH_PATH == 0:
        logger.info("[NPU] 调用 QFA 单算子模式...")
        return _call_npu_fa_op(
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
        )

    # GRAPH_PATH == 7: aclgraph
    # 预处理: list → NPU tensor (必须在编译区域之外完成)
    cu_seqlens_q_t = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(cu_seqlens_kv, dtype=torch.int32).npu()
        if cu_seqlens_kv is not None
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

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    with torch.no_grad():
        torch.npu.synchronize()

        # Network.forward 只调 torch.library op，Python 预处理结果通过参数传入
        fa_args = (
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
            max_seqlen_q,
            max_seqlen_kv,
        )

        # aclgraph: 直接使用 npugraph_ex backend
        logger.info("[NPU] 调用 aclgraph (npugraph_ex)...")
        npu_backend = "npugraph_ex"
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend=npu_backend, dynamic=False
        )
        # mark_static 标记所有 tensor 输入
        for t in (
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
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


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="MXFP8 Flash Attention Golden")
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
    logger.info("MXFP8 Flash Attention Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    logger.info("=" * 60)
    logger.info("场景: %s", "PA" if ENABLE_PA else "TND")
    logger.info("INPUT_LAYOUT=%s, Q_SCALE_LAYOUT=%s", INPUT_LAYOUT, Q_SCALE_LAYOUT)
    logger.info("KV_CACHE_LAYOUT=%s", KV_CACHE_LAYOUT)
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info(
        "ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", _get_seqused_q(), _get_seqused_kv()
    )

    if "gen" in mode:
        logger.info("\n[Step 1] 数据生成")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            qr_bf16,
            kr_bf16,
            block_table_torch,
        ) = generate_data()
        golden_cache.save_input(
            case_name,
            golden_cache.build_input_dict(
                q_fp8,
                k_fp8,
                v_fp8,
                dequant_scale_q,
                dequant_scale_k,
                v_descale,
                p_scale,
                qr_bf16,
                kr_bf16,
                block_table_torch,
            ),
            cache_dir=cdir,
        )
    else:
        logger.info("\n[Step 1] 加载已保存的输入数据")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            qr_bf16,
            kr_bf16,
            block_table_torch,
        ) = golden_cache.load_input(case_name, cache_dir=cdir)

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        exit(0)

    if "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        cpu_out, cpu_lse = cpu_mxfp8_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            _get_seqused_q(),
            _get_seqused_kv(),
            qr_bf16,
            kr_bf16,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("\n[Done] CPU 输出已保存，退出")
        exit(0)

    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        atten_out, lse_out = npu_mxfp8_fa(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            CU_SEQLENS_Q,
            CU_SEQLENS_KV,
            SEQUSED_Q,
            SEQUSED_KV,
            MAX_SEQLEN_Q,
            MAX_SEQLEN_KV,
            block_table_torch,
            qr_bf16,
            kr_bf16,
        )
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("\n[Done] NPU 输出已保存，退出")
        exit(0)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    cpu_tnd_torch = convert_q_bnsd_to_layout(
        cpu_out, _get_seqused_q(), "TND", cu_seqlens=CU_SEQLENS_Q
    )
    result_compare_method.check_result(cpu_tnd_torch, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(
            cpu_lse, _get_seqused_q(), "TND", cu_seqlens=CU_SEQLENS_Q
        )
        # TND padding 位置填 inf 以匹配 NPU 行为：NPU 对超出实际序列长度的 Q 位置输出 inf LSE
        cpu_lse_tnd_torch = fill_tnd_padding(
            cpu_lse_tnd_torch, _get_seqused_q(), CU_SEQLENS_Q, fill_value=float("inf")
        )
        # NPU LSE 输出已改为 N-major 排布 (N, T): N 在外, T 在内
        # CPU golden 经 convert 后是 [T, N, 1] (T-major), 需转成 [N, T] 对齐
        cpu_lse_nt_torch = cpu_lse_tnd_torch.squeeze(-1).permute(1, 0).contiguous()
        result_compare_method.check_result(cpu_lse_nt_torch, lse_out)
