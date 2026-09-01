#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Kernel TestSpec and shared NumPy helpers for QkvRmsNormRopeCacheWithKScale.

The implementation follows the public operator contract, not the AIC/AIV
schedule.  It intentionally keeps the low-precision casts that are observable
at the interface: RoPE output is rounded to BF16 before the dense rotation,
FP8 casts saturate at 448, and M-RoPE INT8 quantization contains the device
path's FP16 intermediate rounding.
"""

from __future__ import annotations

import hashlib
import re
from typing import NamedTuple, Optional, Sequence

import numpy as np

try:
    import ml_dtypes
    from en_dtypes import float8_e8m0
except ImportError as exc:  # pragma: no cover - environment error is explicit
    raise ImportError(
        "QkvRmsNormRopeCacheWithKScale golden requires ml_dtypes and en_dtypes"
    ) from exc


FP8_MAX = np.float32(448.0)
INT8_MAX = np.float32(127.0)
BF16_DTYPE = np.dtype(ml_dtypes.bfloat16)
FP8_DTYPE = np.dtype(ml_dtypes.float8_e4m3fn)
E8M0_DTYPE = np.dtype(float8_e8m0)

# CANN/GE DataType enum values from graph/c_types.h.  q_out_dtype uses these
# integer attributes: 27 = ge::DT_BF16; 36 = ge::DT_FLOAT8_E4M3FN.
GE_DT_BF16 = 27
GE_DT_FLOAT8_E4M3FN = 36


def to_numpy(value):
    """Convert a CPU torch tensor to NumPy without losing BF16/FP8 storage."""

    if value is None or isinstance(value, np.ndarray):
        return value
    if not hasattr(value, "detach"):
        return np.asarray(value)
    tensor = value.detach().cpu().contiguous()
    dtype_name = str(tensor.dtype)
    if "bfloat16" in dtype_name:
        torch = __import__("torch")
        return (
            tensor.view(-1)
            .view(torch.uint16)
            .numpy()
            .view(BF16_DTYPE)
            .reshape(tensor.shape)
        )
    if "float8_e4m3fn" in dtype_name:
        torch = __import__("torch")
        return (
            tensor.view(-1)
            .view(torch.uint8)
            .numpy()
            .view(FP8_DTYPE)
            .reshape(tensor.shape)
        )
    if "float8_e8m0" in dtype_name:
        torch = __import__("torch")
        return (
            tensor.view(-1)
            .view(torch.uint8)
            .numpy()
            .view(E8M0_DTYPE)
            .reshape(tensor.shape)
        )
    return tensor.numpy()


class GoldenResult(NamedTuple):
    q_out: np.ndarray
    q_scale: Optional[np.ndarray]
    k_cache: np.ndarray
    v_cache: np.ndarray
    k_scale_cache: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def to_bf16(value) -> np.ndarray:
    """Round to BF16 (RNE) and retain the BF16 dtype."""

    return np.asarray(value, dtype=np.float32).astype(BF16_DTYPE)


def saturating_fp8_cast(value) -> np.ndarray:
    """Cast to E4M3FN with the operator's finite-value saturation policy."""

    array = np.asarray(value)
    if array.dtype == FP8_DTYPE:
        return array.copy()
    clipped = np.clip(array.astype(np.float32, copy=False), -FP8_MAX, FP8_MAX)
    return clipped.astype(FP8_DTYPE)


def fp8_storage(value) -> np.ndarray:
    """Return E4M3FN storage bytes for exact-rate comparisons."""

    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.copy()
    if array.dtype != FP8_DTYPE:
        array = saturating_fp8_cast(array)
    return np.ascontiguousarray(array).view(np.uint8)


def _dynamic_quant(value: np.ndarray, target: str):
    value = np.asarray(value, dtype=np.float32)
    quant_max = FP8_MAX if target == "fp8" else INT8_MAX
    scale = (np.max(np.abs(value), axis=-1) / quant_max).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = value / scale[..., None]
    if target == "fp8":
        quantized = saturating_fp8_cast(normalized)
        # DAV_3510 F32->E4M3FN with SatMode::SAT and CTRL[50]=0 writes raw
        # zero for NaN.  This includes explicit NaN/Inf inputs and the NaN
        # produced by an all-zero row's 0/0 normalization.
        raw = quantized.view(np.uint8)
        raw[~np.isfinite(normalized)] = np.uint8(0)
    else:
        rounded = np.rint(normalized.astype(np.float16).astype(np.float32))
        quantized = np.clip(rounded, -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scale


def _mx_quant_cublas(value: np.ndarray):
    """Quantize D32 blocks to E4M3FN with cuBLAS-compatible E8M0 scales."""

    shape = value.shape
    blocks = np.asarray(value, dtype=np.float32).reshape(
        *shape[:-1], shape[-1] // 32, 32
    )
    amax = np.max(np.abs(blocks), axis=-1)
    nonfinite_amax = ~np.isfinite(amax)
    scaled_amax = (amax / FP8_MAX).astype(np.float32)
    bits = scaled_amax.view(np.uint32)
    biased = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int16)
    mantissa = bits & np.uint32(0x007FFFFF)
    round_up = ((biased > 0) & (biased < 254) & (mantissa > 0)) | (
        (biased == 0) & (mantissa > (1 << 22))
    )
    exponent = biased + round_up.astype(np.int16) - 127
    exponent = np.where(amax == 0, -127, exponent)
    exponent = np.maximum(exponent, -127)
    exponent = np.where(exponent > 127, 128, exponent).astype(np.int16)

    scale = np.exp2(np.minimum(exponent, 127).astype(np.float32))[..., None]
    scale = np.where(nonfinite_amax[..., None], np.float32(np.nan), scale)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quantized = saturating_fp8_cast(blocks / scale)
    quantized_raw = np.where(
        nonfinite_amax[..., None], np.uint8(0), quantized.view(np.uint8)
    )
    quantized = quantized_raw.reshape(shape).view(FP8_DTYPE)

    scale_raw = np.where(nonfinite_amax | (exponent > 127), 255, exponent + 127).astype(
        np.uint8
    )
    scales = scale_raw.view(E8M0_DTYPE).reshape(*shape[:-1], shape[-1] // 32)
    return quantized, scales


def _normal_rope_positions(token_num: int, query_start_loc, seq_lens):
    starts = np.asarray(query_start_loc, dtype=np.int64)
    lengths = np.asarray(seq_lens, dtype=np.int64)
    _require(
        starts.ndim == 1 and starts.size >= 2,
        "query_start_loc must have shape [Batch+1]",
    )
    _require(
        lengths.ndim == 1 and lengths.size + 1 == starts.size,
        "seq_lens must have shape [Batch]",
    )
    _require(
        starts[0] == 0 and starts[-1] == token_num,
        "query_start_loc must start at 0 and end at T",
    )
    _require(
        bool(np.all(starts[1:] >= starts[:-1])), "query_start_loc must be nondecreasing"
    )

    positions = np.empty(token_num, dtype=np.int64)
    for batch in range(lengths.size):
        begin, end = int(starts[batch]), int(starts[batch + 1])
        current = end - begin
        first = int(lengths[batch]) - current
        _require(
            first >= 0, "seq_lens values must be at least the number of current tokens"
        )
        positions[begin:end] = first + np.arange(current, dtype=np.int64)
    return positions


def _select_cos_sin(
    cos_sin,
    token_num: int,
    query_start_loc,
    seq_lens,
    mrope_position,
    mrope_section: Optional[Sequence[int]],
):
    table = np.asarray(cos_sin, dtype=np.float32)
    head_dim = int(table.shape[1])
    half_dim = head_dim // 2
    has_position = mrope_position is not None
    has_section = mrope_section is not None and len(mrope_section) != 0
    _require(
        has_position == has_section,
        "mrope_position and non-empty mrope_section must be provided together",
    )

    if not has_position:
        _require(
            query_start_loc is not None and seq_lens is not None,
            "RoPE requires query_start_loc and seq_lens",
        )
        positions = _normal_rope_positions(token_num, query_start_loc, seq_lens)
        _require(
            bool(np.all(positions < table.shape[0])),
            "RoPE position exceeds cos_sin rows",
        )
        selected = table[positions]
        return selected[:, :half_dim], selected[:, half_dim:], False

    _require(
        query_start_loc is None and seq_lens is None,
        "M-RoPE requires query_start_loc=None and seq_lens=None",
    )
    section_t, section_h, section_w = (int(item) for item in mrope_section)
    _require(
        min(section_t, section_h, section_w) >= 0,
        "mrope_section values must be nonnegative",
    )
    _require(
        section_t + section_h + section_w <= half_dim,
        "sum(mrope_section) must not exceed D/2",
    )
    _require(
        section_h <= 21 and section_w <= 21,
        "for D=128, M-RoPE H/W sections must not exceed 21",
    )

    position = np.asarray(mrope_position, dtype=np.int64)
    _require(position.shape == (token_num, 3), "mrope_position must have shape [T,3]")
    _require(
        bool(np.all(position >= 0) and np.all(position < table.shape[0])),
        "M-RoPE position exceeds cos_sin rows",
    )

    raw = table[position]  # [T, 3, D]
    lane = np.arange(half_dim, dtype=np.int64)
    group = lane // 3
    axis = np.zeros(half_dim, dtype=np.int64)
    axis[(lane % 3 == 1) & (group < section_h)] = 1
    axis[(lane % 3 == 2) & (group < section_w)] = 2
    token = np.arange(token_num, dtype=np.int64)[:, None]
    cos = raw[token, axis[None, :], lane[None, :]]
    sin = raw[token, axis[None, :], (half_dim + lane)[None, :]]
    return cos, sin, True


def _apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    half_dim = value.shape[-1] // 2
    low, high = value[..., :half_dim], value[..., half_dim:]
    return np.concatenate(
        (
            low * cos[:, None, :] - high * sin[:, None, :],
            high * cos[:, None, :] + low * sin[:, None, :],
        ),
        axis=-1,
    )


def _normalize_dtype_name(value) -> str:
    if value is None:
        return "float8_e4m3fn"
    if isinstance(value, (int, np.integer)):
        return {
            GE_DT_BF16: "bfloat16",
            GE_DT_FLOAT8_E4M3FN: "float8_e4m3fn",
        }.get(int(value), str(value))
    return str(value).lower().replace("torch.", "").replace("dt_", "")


def qkv_rms_norm_rope_cache_with_k_scale_numpy(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    rotation,
    v_scale,
    mrope_position,
    *,
    head_nums,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=1e-6,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=None,
) -> GoldenResult:
    """Compute all logical outputs without modifying the input caches."""

    _require(
        head_nums is not None and len(head_nums) == 3,
        "head_nums must contain [Nq,Nk,Nv]",
    )
    q_heads, k_heads, v_heads = (int(item) for item in head_nums)
    is_mx = str(q_quant_mode) == "Mx" and str(k_quant_mode) == "Mx"
    _require(0 < q_heads <= 64, "Nq must satisfy 0 < Nq <= 64")
    _require(q_heads == 8 * k_heads, "Nq must equal 8 * Nk")
    _require(k_heads == v_heads, "Nk must equal Nv")

    in_layout = "TND" if layout_qkv in (None, "") else str(layout_qkv).upper()
    out_layout = "NTD" if layout_q_out in (None, "") else str(layout_q_out).upper()
    _require(in_layout in ("TND", "NTD"), "layout_qkv must be TND or NTD")
    _require(out_layout in ("TND", "NTD"), "layout_q_out must be TND or NTD")
    _require(
        not (in_layout == "NTD" and out_layout == "TND"),
        "NTD input with TND output is unsupported",
    )

    qkv_array = np.asarray(qkv)
    _require(qkv_array.ndim == 3, "qkv must be rank 3")
    token_num = int(qkv_array.shape[0] if in_layout == "TND" else qkv_array.shape[1])
    total_heads = int(qkv_array.shape[1] if in_layout == "TND" else qkv_array.shape[0])
    head_dim = int(qkv_array.shape[2])
    _require(head_dim == 128, "D must be 128")
    _require(
        total_heads == q_heads + k_heads + v_heads,
        "qkv head dimension does not match head_nums",
    )
    _require(np.asarray(q_gamma).shape == (head_dim,), "q_gamma must have shape [D]")
    _require(np.asarray(k_gamma).shape == (head_dim,), "k_gamma must have shape [D]")
    _require(
        np.asarray(cos_sin).ndim == 2 and np.asarray(cos_sin).shape[1] == head_dim,
        "cos_sin must have shape [MaxSeqLen,D]",
    )
    _require(
        np.asarray(slot_mapping).shape == (token_num,),
        "slot_mapping must have shape [T]",
    )
    if not is_mx:
        _require(
            rotation is not None and np.asarray(rotation).shape == (head_dim, head_dim),
            "rotation must have shape [D,D]",
        )

    k_cache_array = np.asarray(k_cache)
    v_cache_array = np.asarray(v_cache)
    k_scale_array = np.asarray(k_scale_cache)
    _require(k_cache_array.ndim == 4, "k_cache must be rank 4")
    block_num, cache_k_heads, block_size, cache_dim = k_cache_array.shape
    _require(
        (cache_k_heads, cache_dim) == (k_heads, head_dim),
        "k_cache must have shape [BlockNum,Nk,BlockSize,D]",
    )
    _require(
        v_cache_array.shape == (block_num, v_heads, block_size, head_dim),
        "v_cache must have shape [BlockNum,Nv,BlockSize,D]",
    )
    expected_k_scale_shape = (
        (block_num, k_heads, block_size, head_dim // 32)
        if is_mx
        else (block_num, k_heads, block_size, 1)
    )
    _require(
        k_scale_array.shape == expected_k_scale_shape,
        f"k_scale_cache must have shape {expected_k_scale_shape}",
    )

    def rms_norm(value, gamma):
        mean_square = np.sum(value * value, axis=-1, keepdims=True) / np.float32(
            head_dim
        )
        return (
            value
            / np.sqrt(mean_square + np.float32(epsilon))
            * np.asarray(gamma, dtype=np.float32)
        )

    cos, sin, is_mrope = _select_cos_sin(
        cos_sin, token_num, query_start_loc, seq_lens, mrope_position, mrope_section
    )
    rotation_fp32 = None if is_mx else to_bf16(rotation).astype(np.float32)
    dtype_name = _normalize_dtype_name(q_out_dtype)
    if is_mx:
        _require(is_mrope, "Mx requires M-RoPE inputs")
        _require(
            "float8_e4m3fn" in dtype_name or dtype_name == str(GE_DT_FLOAT8_E4M3FN),
            "Mx requires FP8 E4M3FN q_out",
        )
        _require(
            np.asarray(v_scale).shape == (v_heads, head_dim),
            "M-RoPE MX v_scale must have shape [Nv,D]",
        )
        v_multiplier = np.asarray(v_scale, dtype=np.float32)[None, :, :]
        q_dtype = FP8_DTYPE
    elif is_mrope:
        _require(str(q_quant_mode) == "NoQuant", "M-RoPE requires q_quant_mode=NoQuant")
        _require(
            str(k_quant_mode) == "PerTokenPerHead",
            "M-RoPE requires k_quant_mode=PerTokenPerHead",
        )
        _require(
            "bfloat16" in dtype_name or dtype_name in ("bf16", str(GE_DT_BF16)),
            "M-RoPE requires BF16 q_out",
        )
        expected_v_scale = (v_heads, head_dim)
        _require(
            np.asarray(v_scale).shape == expected_v_scale,
            "M-RoPE v_scale must have shape [Nv,D]",
        )
        v_multiplier = np.asarray(v_scale, dtype=np.float32)[None, :, :]
        q_dtype = BF16_DTYPE
    else:
        _require(
            str(q_quant_mode) == "PerTokenPerHead",
            "RoPE requires q_quant_mode=PerTokenPerHead",
        )
        _require(
            str(k_quant_mode) == "PerTokenPerHead",
            "RoPE requires k_quant_mode=PerTokenPerHead",
        )
        _require(
            "float8_e4m3fn" in dtype_name or dtype_name == str(GE_DT_FLOAT8_E4M3FN),
            "RoPE requires FP8 E4M3FN q_out",
        )
        _require(
            np.asarray(v_scale).shape == (v_heads,), "RoPE v_scale must have shape [Nv]"
        )
        v_multiplier = np.asarray(v_scale, dtype=np.float32)[None, :, None]
        q_dtype = FP8_DTYPE

    q_shape = (
        (q_heads, token_num, head_dim)
        if out_layout == "NTD"
        else (token_num, q_heads, head_dim)
    )
    q_out = np.empty(q_shape, dtype=q_dtype)
    if is_mx:
        q_scale = np.empty(q_shape[:-1] + (head_dim // 32,), dtype=E8M0_DTYPE)
    elif is_mrope:
        q_scale = None
    else:
        q_scale = np.empty(q_shape[:-1], dtype=np.float32)
    k_cache_out = k_cache_array.copy()
    v_cache_out = v_cache_array.copy()
    k_scale_cache_out = (
        k_scale_array.copy() if is_mx else k_scale_array.astype(np.float32, copy=True)
    )
    slots = np.asarray(slot_mapping, dtype=np.int64)
    _require(
        bool(np.all(slots >= 0) and np.all(slots < block_num * block_size)),
        "slot_mapping exceeds cache capacity",
    )

    # Avoid materializing a full FP32 QKV plus all normalization/RoPE/Cube
    # intermediates.  At T=262143 and Nq=64 those temporaries exceed tens of
    # GiB.  The operator is token-independent before cache scatter, so chunking
    # preserves the exact element-wise formula and output order.
    chunk_tokens = 4096
    for begin in range(0, token_num, chunk_tokens):
        end = min(token_num, begin + chunk_tokens)
        if in_layout == "NTD":
            qkv_chunk = np.transpose(
                qkv_array[:, begin:end, :].astype(np.float32), (1, 0, 2)
            )
        else:
            qkv_chunk = qkv_array[begin:end].astype(np.float32)
        q = qkv_chunk[:, :q_heads, :]
        k = qkv_chunk[:, q_heads : q_heads + k_heads, :]
        v = qkv_chunk[:, q_heads + k_heads :, :]

        q_rope = _apply_rope(rms_norm(q, q_gamma), cos[begin:end], sin[begin:end])
        k_rope = _apply_rope(rms_norm(k, k_gamma), cos[begin:end], sin[begin:end])
        if is_mx:
            q_rot = q_rope
            k_rot = k_rope
            q_chunk, q_scale_chunk = _mx_quant_cublas(q_rot)
            k_quant, k_scale = _mx_quant_cublas(k_rot)
        else:
            q_rot = to_bf16(q_rope).astype(np.float32) @ rotation_fp32
            k_rot = to_bf16(k_rope).astype(np.float32) @ rotation_fp32

        if is_mrope and not is_mx:
            q_chunk = to_bf16(q_rot)
            k_quant, k_scale = _dynamic_quant(k_rot, "int8")
        elif not is_mx:
            q_chunk, q_scale_chunk = _dynamic_quant(q_rot, "fp8")
            k_quant, k_scale = _dynamic_quant(k_rot, "fp8")
        v_quant = saturating_fp8_cast(v * v_multiplier)

        if out_layout == "NTD":
            q_out[:, begin:end, :] = np.transpose(q_chunk, (1, 0, 2))
            if q_scale is not None:
                axes = (1, 0, 2) if is_mx else (1, 0)
                q_scale[:, begin:end] = np.transpose(q_scale_chunk, axes)
        else:
            q_out[begin:end] = q_chunk
            if q_scale is not None:
                q_scale[begin:end] = q_scale_chunk

        chunk_slots = slots[begin:end]
        block_ids = chunk_slots // block_size
        block_offsets = chunk_slots % block_size
        k_cache_out[block_ids, :, block_offsets, :] = k_quant
        v_cache_out[block_ids, :, block_offsets, :] = v_quant
        if is_mx:
            k_scale_cache_out[block_ids, :, block_offsets, :] = k_scale
        else:
            k_scale_cache_out[block_ids, :, block_offsets, 0] = k_scale

    return GoldenResult(q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out)


# ============================================================================
# Shared deterministic input generation
# ============================================================================


def _seed(testcase_name) -> int:
    digest = hashlib.sha256(str(testcase_name or "qkv").encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


_FULL_CASE_META = re.compile(
    r"__meta_b(?P<batch>\d+)_h(?P<history>[01])_s(?P<seed>\d+)_"
    r"p(?P<slot>linear|reverse|random|affine)_rk(?P<k_pad>\d+)_"
    r"rv(?P<v_pad>\d+)_rs(?P<scale_pad>\d+)$"
)


def _full_case_meta(testcase_name):
    match = _FULL_CASE_META.search(str(testcase_name or ""))
    if match is None:
        return None
    values = match.groupdict()
    return {
        "batch": int(values["batch"]),
        "history": bool(int(values["history"])),
        "seed": int(values["seed"]),
        "slot": values["slot"],
    }


def _write_back(target, value):
    if target is None:
        return None
    value = np.asarray(value)
    if isinstance(target, np.ndarray):
        target[...] = value.astype(target.dtype, copy=False)
        return target

    import torch

    dtype_name = str(target.dtype)
    if "bfloat16" in dtype_name:
        source = torch.from_numpy(
            np.ascontiguousarray(value.astype(BF16_DTYPE)).view(np.uint16)
        ).view(torch.bfloat16)
    elif "float8_e4m3fn" in dtype_name:
        source = torch.from_numpy(
            np.ascontiguousarray(value.astype(FP8_DTYPE)).view(np.uint8)
        ).view(torch.float8_e4m3fn)
    elif "float8_e8m0" in dtype_name:
        source = torch.from_numpy(
            np.ascontiguousarray(value.astype(E8M0_DTYPE)).view(np.uint8)
        ).view(torch.float8_e8m0fnu)
    else:
        source = torch.from_numpy(np.ascontiguousarray(value)).to(dtype=target.dtype)
    target.copy_(source.to(target.device).reshape(target.shape))
    return target


def _write_stress_qkv(target, token_num, rng):
    """Fill a full-matrix QKV tensor without a full-size FP32 temporary."""

    is_tnd = int(target.shape[0]) == token_num
    for begin in range(0, token_num, 2048):
        end = min(token_num, begin + 2048)
        view = target[begin:end] if is_tnd else target[:, begin:end, :]
        values = rng.standard_normal(tuple(view.shape), dtype=np.float32)
        values *= np.float32(0.45)
        _write_back(view, to_bf16(values))


def _prepare(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    rotation,
    v_scale,
    mrope_position,
    *,
    testcase_name=None,
    head_nums=None,
    layout_qkv="TND",
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
):
    full_meta = _full_case_meta(testcase_name)
    rng = np.random.default_rng(
        full_meta["seed"] if full_meta is not None else _seed(testcase_name)
    )
    qkv_shape = tuple(qkv.shape)
    token_num = int(slot_mapping.shape[0])
    head_dim = int(qkv_shape[-1])
    block_num, _, block_size, _ = tuple(k_cache.shape)
    is_mx = str(q_quant_mode) == "Mx" and str(k_quant_mode) == "Mx"

    # The full-800 stress matrix leaves QKV/cache storage as generated by TTK.
    # Re-materializing a largest-case QKV tensor here would add a >10 GiB FP32
    # temporary before Golden generation.  Small regression cases retain their
    # original deterministic standalone data generation.
    qkv_data = (
        None
        if full_meta is not None
        else to_bf16(rng.normal(0.0, 0.45, size=qkv_shape).astype(np.float32))
    )
    q_gamma_data = rng.uniform(0.75, 1.25, size=q_gamma.shape).astype(np.float32)
    k_gamma_data = rng.uniform(0.75, 1.25, size=k_gamma.shape).astype(np.float32)

    rows = int(cos_sin.shape[0])
    half = head_dim // 2
    positions = np.arange(rows, dtype=np.float32)[:, None]
    frequencies = np.exp(-np.arange(half, dtype=np.float32)[None, :] / np.float32(half))
    angles = positions * frequencies * np.float32(0.03125)
    cos_sin_data = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(
        np.float32
    )

    name = str(testcase_name or "")
    if full_meta is None and head_nums is not None:
        q_heads, k_heads, _ = (int(item) for item in head_nums)
        qk_heads = q_heads + k_heads
        layout = str(layout_qkv or "TND").upper()
        if is_mx:
            lane = np.arange(head_dim, dtype=np.int32)
            magnitude = np.asarray([2.0**-5, 2.0**-1, 2.0**3, 2.0**7], np.float32)[
                lane // 32
            ]
            row = ((lane * 13 + 7) % 29).astype(np.float32) / 29.0 + 0.5
            row *= magnitude * np.where(lane & 1, -1.0, 1.0)
            total_heads = q_heads + k_heads + int(head_nums[2])
            head = (1.0 + (np.arange(total_heads, dtype=np.float32) % 5) / 64.0)[
                None, :, None
            ]
            token_scale = (1.0 + (np.arange(token_num, dtype=np.float32) % 7) / 32.0)[
                :, None, None
            ]
            qkv_tnd = token_scale * head * row[None, None, :]
            qkv_data = to_bf16(
                np.transpose(qkv_tnd, (1, 0, 2)) if layout == "NTD" else qkv_tnd
            )

        qkv_profile = np.asarray(qkv_data, dtype=np.float32).copy()
        if "nan_qk" in name or "inf_qk" in name:
            special = np.float32(np.nan if "nan_qk" in name else np.inf)
            if layout == "NTD":
                qkv_profile[:qk_heads, 0, 0] = special
            else:
                qkv_profile[0, :qk_heads, 0] = special
            qkv_data = to_bf16(qkv_profile)
        elif "zero_qk" in name:
            if layout == "NTD":
                qkv_profile[:qk_heads, :, :] = 0
            else:
                qkv_profile[:, :qk_heads, :] = 0
            qkv_data = to_bf16(qkv_profile)
        elif "scale_edge_qk" in name or "max_finite_qk" in name:
            if layout == "NTD":
                qkv_profile[:qk_heads, :, :] = 1
            else:
                qkv_profile[:, :qk_heads, :] = 1
            qkv_data = to_bf16(qkv_profile)

        lane = np.arange(head_dim, dtype=np.int32)
        if "scale_order" in name:
            q_gamma_data = np.asarray([2.0**-6, 2.0**-2, 2.0**2, 2.0**6], np.float32)[
                lane // 32
            ]
            k_gamma_data = q_gamma_data[::-1].copy()
        elif "scale_edge_qk" in name:
            edge = np.asarray(
                [0x04600000, 0x04600001, 0x04600002, 0x3F800000], np.uint32
            ).view(np.float32)
            q_gamma_data = edge[lane // 32]
            k_gamma_data = q_gamma_data[::-1].copy()
        elif "max_finite_qk" in name:
            q_gamma_data.fill(np.asarray([0x7F7FFFFF], np.uint32).view(np.float32)[0])
            k_gamma_data = q_gamma_data.copy()
        elif "tiny_qk" in name:
            q_gamma_data.fill(np.float32(2.0**-140))
            k_gamma_data = q_gamma_data.copy()
        elif "saturation_qk" in name:
            q_gamma_data.fill(np.float32(2.0**120))
            k_gamma_data = q_gamma_data.copy()

        if any(
            tag in name for tag in ("scale_order", "scale_edge_qk", "max_finite_qk")
        ):
            cos_sin_data[:, :half] = np.float32(1.0)
            cos_sin_data[:, half:] = np.float32(0.0)

    capacity = block_num * block_size
    token = np.arange(token_num, dtype=np.int64)
    if full_meta is None:
        slots = ((token * 7 + 3) % capacity).astype(np.int32)
        if np.unique(slots).size != token_num:
            slots = rng.choice(capacity, size=token_num, replace=False).astype(np.int32)
    elif full_meta["slot"] == "linear":
        slots = token.astype(np.int32)
    elif full_meta["slot"] == "reverse":
        slots = np.arange(
            capacity - 1, capacity - token_num - 1, -1, dtype=np.int64
        ).astype(np.int32)
    elif full_meta["slot"] == "random":
        slots = rng.permutation(capacity)[:token_num].astype(np.int32)
    else:
        slots = ((token * (capacity - 1) + 17) % capacity).astype(np.int32)

    def _move_slots_to_front(values):
        nonlocal slots
        values = [int(value) for value in values if int(value) < capacity]
        tail = [int(value) for value in slots if int(value) not in values]
        slots = np.asarray(values + tail, dtype=np.int32)[:token_num]

    if "slot_block_begin" in name:
        _move_slots_to_front([0])
    elif "slot_block_end" in name:
        _move_slots_to_front([block_size - 1])
    elif "slot_cross_block" in name and token_num >= 2:
        _move_slots_to_front([block_size - 1, block_size])

    if full_meta is None:
        if np.dtype(to_numpy(k_cache).dtype) == np.dtype(np.int8):
            k_cache_data = rng.integers(-31, 32, size=k_cache.shape, dtype=np.int8)
        else:
            k_cache_data = saturating_fp8_cast(
                rng.uniform(-4.0, 4.0, size=k_cache.shape).astype(np.float32)
            )
        v_cache_data = saturating_fp8_cast(
            rng.uniform(-4.0, 4.0, size=v_cache.shape).astype(np.float32)
        )
        if np.dtype(to_numpy(k_scale_cache).dtype) == E8M0_DTYPE:
            k_scale_raw = rng.integers(1, 253, size=k_scale_cache.shape, dtype=np.uint8)
            k_scale_data = k_scale_raw.view(E8M0_DTYPE)
        else:
            k_scale_data = rng.uniform(0.01, 0.5, size=k_scale_cache.shape).astype(
                np.float32
            )
    else:
        k_cache_data = v_cache_data = k_scale_data = None

    if query_start_loc is not None:
        batch = int(seq_lens.shape[0])
        if full_meta is not None:
            base, remainder = divmod(token_num, batch)
            current = [base + (index < remainder) for index in range(batch)]
        elif batch == 1:
            current = [token_num]
        elif batch == 2:
            first = max(1, token_num // 3)
            current = [first, token_num - first]
        else:
            base, remainder = divmod(token_num, batch)
            current = [base + (index < remainder) for index in range(batch)]
        starts = np.concatenate(([0], np.cumsum(current))).astype(np.int32)
        if full_meta is not None and full_meta["history"]:
            histories = (
                257 + 113 * np.arange(batch, dtype=np.int32) + full_meta["seed"] % 97
            )
        elif full_meta is not None:
            histories = np.zeros(batch, dtype=np.int32)
        else:
            histories = np.arange(1, batch + 1, dtype=np.int32) * 3
        seq_data = np.asarray(current, dtype=np.int32) + histories
        _write_back(query_start_loc, starts)
        _write_back(seq_lens, seq_data)

    rotation_data = None
    if rotation is not None:
        rotation_data = to_bf16(
            np.eye(head_dim, dtype=np.float32)
            + rng.normal(0.0, 0.01, size=rotation.shape).astype(np.float32)
        )
    v_scale_data = (
        None
        if v_scale is None
        else rng.uniform(0.5, 1.5, size=v_scale.shape).astype(np.float32)
    )
    if mrope_position is not None:
        token = np.arange(token_num, dtype=np.int64)
        position_data = np.stack(
            ((token * 3) % rows, (token * 5 + 1) % rows, (token * 7 + 2) % rows),
            axis=1,
        ).astype(np.int32)
        if "position_1024" in name:
            boundary = np.asarray(
                [[1023, 1024, 1025], [1024, 1025, 1023], [1025, 1023, 1024]],
                dtype=np.int32,
            )
            position_data[:] = boundary[np.arange(token_num) % len(boundary)]
        _write_back(mrope_position, position_data)

    for target, value in (
        (qkv, qkv_data),
        (q_gamma, q_gamma_data),
        (k_gamma, k_gamma_data),
        (cos_sin, cos_sin_data),
        (slot_mapping, slots),
        (k_cache, k_cache_data),
        (v_cache, v_cache_data),
        (k_scale_cache, k_scale_data),
        (rotation, rotation_data),
        (v_scale, v_scale_data),
    ):
        if value is not None:
            _write_back(target, value)
    if full_meta is not None:
        _write_stress_qkv(qkv, token_num, rng)


def kernel_customize_inputs(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc=None,
    seq_lens=None,
    rotation=None,
    v_scale=None,
    mrope_position=None,
    **kwargs,
):
    _prepare(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        rotation,
        v_scale,
        mrope_position,
        testcase_name=kwargs.get("testcase_name"),
        head_nums=kwargs.get("head_nums"),
        layout_qkv=kwargs.get("layout_qkv", "TND"),
        q_quant_mode=kwargs.get("q_quant_mode", "PerTokenPerHead"),
        k_quant_mode=kwargs.get("k_quant_mode", "PerTokenPerHead"),
    )
    return (
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        rotation,
        v_scale,
        mrope_position,
    )


# ============================================================================
# Shared precision comparison
# ============================================================================

BF16_RTOL = 1e-3
BF16_ATOL = 1e-8
BF16_PTOL = 1e-3
FP8_RTOL = 1e-2
FP8_ATOL = 1e-8
FP8_PTOL = 1e-2
SCALE_FP32_RTOL = 1e-3
SCALE_FP32_ATOL = 1e-8
SCALE_FP32_PTOL = 1e-3
INT8_MAX_ABS_DIFF = 1
PTA_FUNCTIONAL_API = "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale"
PTA_INPLACE_API = "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_"

TOLERANCE = {
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
    # Custom comparison owns the FP8/INT8 metrics.  These valid fallback
    # standards are still needed while TTK resolves the pre-hook tolerance.
    "float8_e4m3fn": {"standard": "binary_equal"},
    "float8_e8m0": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


def _rate(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return 1.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


def _exact_rate(actual, expected) -> float:
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    if (
        actual.shape == expected.shape
        and actual.dtype.itemsize == expected.dtype.itemsize
    ):
        itemsize = actual.dtype.itemsize
        actual_bytes = actual.view(np.uint8).reshape(actual.shape + (itemsize,))
        expected_bytes = expected.view(np.uint8).reshape(expected.shape + (itemsize,))
        return _rate(np.all(actual_bytes == expected_bytes, axis=-1))
    return _rate(actual.astype(np.float32) == expected.astype(np.float32))


def _close_rate(actual, expected, rtol, atol) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    matched = 0
    chunk_elements = 1 << 20
    for begin in range(0, actual_flat.size, chunk_elements):
        end = min(actual_flat.size, begin + chunk_elements)
        matched += int(
            np.count_nonzero(
                np.isclose(
                    np.asarray(actual_flat[begin:end], dtype=np.float32),
                    np.asarray(expected_flat[begin:end], dtype=np.float32),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            )
        )
    return 1.0 if actual_flat.size == 0 else matched / actual_flat.size


def _abs_le_rate(actual, expected, limit) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    matched = 0
    chunk_elements = 1 << 20
    for begin in range(0, actual_flat.size, chunk_elements):
        end = min(actual_flat.size, begin + chunk_elements)
        difference = np.abs(
            actual_flat[begin:end].astype(np.int16, copy=False)
            - expected_flat[begin:end].astype(np.int16, copy=False)
        )
        matched += int(np.count_nonzero(difference <= limit))
    return 1.0 if actual_flat.size == 0 else matched / actual_flat.size


def _cache_rows(cache, slots):
    cache = np.asarray(cache)
    block_size = cache.shape[2]
    slots = np.asarray(slots, dtype=np.int64)
    return cache[slots // block_size, :, slots % block_size, ...]


def _untouched_exact(actual, expected, slots) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    mask = np.ones((actual.shape[0], actual.shape[2]), dtype=bool)
    block_size = actual.shape[2]
    for slot in slots:
        mask[int(slot) // block_size, int(slot) % block_size] = False
    actual_rows = np.transpose(actual, (0, 2, 1, 3))[mask]
    expected_rows = np.transpose(expected, (0, 2, 1, 3))[mask]
    return _exact_rate(actual_rows, expected_rows)


def compare_outputs(*values, compare_context):
    output_count = len(values) // 2
    # Kernel hooks receive NumPy arrays, while ACLNN hooks may receive CPU
    # Torch tensors.  Normalize here because Torch cannot expose BF16/FP8
    # tensors through Tensor.numpy() directly.
    outputs = [to_numpy(value) for value in values[:output_count]]
    goldens = [to_numpy(value) for value in values[output_count:]]
    attrs = dict(compare_context.attributes)
    api_name = str(compare_context.api_name)
    if api_name.startswith("torch.ops."):
        api_name = api_name[len("torch.ops.") :]
    q_quant_mode = attrs.get("q_quant_mode", attrs.get("qQuantMode", "PerTokenPerHead"))
    k_quant_mode = attrs.get("k_quant_mode", attrs.get("kQuantMode", "PerTokenPerHead"))
    is_mx = str(q_quant_mode) == "Mx" and str(k_quant_mode) == "Mx"
    is_mrope = str(q_quant_mode) == "NoQuant"
    is_kernel = api_name == "qkv_rms_norm_rope_cache_with_k_scale"
    is_pta = api_name in (PTA_FUNCTIONAL_API, PTA_INPLACE_API)
    is_functional = api_name == PTA_FUNCTIONAL_API

    # Only slot_mapping is needed from the inputs.  Converting every input here
    # would duplicate multi-GiB QKV/cos tensors in the full-800 matrix.
    inputs = compare_context.input_tensors
    slots = np.asarray(to_numpy(inputs[4]), dtype=np.int64)

    if is_mx:
        q_actual, qs_actual, k_actual, v_actual, ks_actual = outputs[:5]
        q_golden, qs_golden, k_golden, v_golden, ks_golden = goldens[:5]
    elif is_mrope and (is_kernel or is_pta):
        q_actual, _, k_actual, v_actual, ks_actual = outputs[:5]
        q_golden, _, k_golden, v_golden, ks_golden = goldens[:5]
    elif is_mrope:
        q_actual, k_actual, v_actual, ks_actual = outputs[:4]
        q_golden, k_golden, v_golden, ks_golden = goldens[:4]
    else:
        q_actual, qs_actual, k_actual, v_actual, ks_actual = outputs[:5]
        q_golden, qs_golden, k_golden, v_golden, ks_golden = goldens[:5]

    k_rows = _cache_rows(k_actual, slots)
    k_golden_rows = _cache_rows(k_golden, slots)
    v_rows = _cache_rows(v_actual, slots)
    v_golden_rows = _cache_rows(v_golden, slots)
    ks_rows = _cache_rows(ks_actual, slots)
    ks_golden_rows = _cache_rows(ks_golden, slots)

    metrics = {}
    if is_mx:
        metrics.update(
            {
                "q_fp8_close": _close_rate(q_actual, q_golden, FP8_RTOL, FP8_ATOL),
                "q_scale_e8m0_exact": _exact_rate(qs_actual, qs_golden),
                "k_fp8_close": _close_rate(k_rows, k_golden_rows, FP8_RTOL, FP8_ATOL),
                "k_scale_e8m0_exact": _exact_rate(ks_rows, ks_golden_rows),
                "v_fp8_close": _close_rate(v_rows, v_golden_rows, FP8_RTOL, FP8_ATOL),
            }
        )
        thresholds = {
            "q_fp8_close": 1.0 - FP8_PTOL,
            "q_scale_e8m0_exact": 1.0,
            "k_fp8_close": 1.0 - FP8_PTOL,
            "k_scale_e8m0_exact": 1.0,
            "v_fp8_close": 1.0 - FP8_PTOL,
        }
    elif is_mrope:
        metrics.update(
            {
                "q_bf16_close": _close_rate(q_actual, q_golden, BF16_RTOL, BF16_ATOL),
                "k_int8_abs_le_1": _abs_le_rate(
                    k_rows, k_golden_rows, INT8_MAX_ABS_DIFF
                ),
                "k_scale_fp32_close": _close_rate(
                    ks_rows, ks_golden_rows, SCALE_FP32_RTOL, SCALE_FP32_ATOL
                ),
                "v_fp8_close": _close_rate(v_rows, v_golden_rows, FP8_RTOL, FP8_ATOL),
            }
        )
        thresholds = {
            "q_bf16_close": 1.0 - BF16_PTOL,
            "k_int8_abs_le_1": 1.0,
            "k_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "v_fp8_close": 1.0 - FP8_PTOL,
        }
    else:
        metrics.update(
            {
                "q_fp8_close": _close_rate(q_actual, q_golden, FP8_RTOL, FP8_ATOL),
                "q_scale_fp32_close": _close_rate(
                    qs_actual, qs_golden, SCALE_FP32_RTOL, SCALE_FP32_ATOL
                ),
                "k_fp8_close": _close_rate(k_rows, k_golden_rows, FP8_RTOL, FP8_ATOL),
                "k_scale_fp32_close": _close_rate(
                    ks_rows, ks_golden_rows, SCALE_FP32_RTOL, SCALE_FP32_ATOL
                ),
                "v_fp8_close": _close_rate(v_rows, v_golden_rows, FP8_RTOL, FP8_ATOL),
            }
        )
        thresholds = {
            "q_fp8_close": 1.0 - FP8_PTOL,
            "q_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "k_fp8_close": 1.0 - FP8_PTOL,
            "k_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "v_fp8_close": 1.0 - FP8_PTOL,
        }

    metrics.update(
        {
            "k_cache_untouched": _untouched_exact(k_actual, k_golden, slots),
            "v_cache_untouched": _untouched_exact(v_actual, v_golden, slots),
            "k_scale_cache_untouched": _untouched_exact(ks_actual, ks_golden, slots),
        }
    )
    thresholds.update(
        {
            "k_cache_untouched": 1.0,
            "v_cache_untouched": 1.0,
            "k_scale_cache_untouched": 1.0,
        }
    )

    if is_functional:
        original_actual = outputs[-3:]
        original_golden = goldens[-3:]
        for name, actual, expected in zip(
            ("functional_k_input", "functional_v_input", "functional_k_scale_input"),
            original_actual,
            original_golden,
        ):
            metrics[name] = _exact_rate(actual, expected)
            thresholds[name] = 1.0

    failures = [
        f"{name}={metrics[name]:.6f} < {threshold:.6f}"
        for name, threshold in thresholds.items()
        if metrics[name] + 1e-12 < threshold
    ]
    precision = "; ".join(
        f"{name}={value * 100:.6f}%" for name, value in metrics.items()
    )
    return {
        "pass": not failures,
        "precision": precision,
        "error_info": "; ".join(failures),
        "metrics": metrics,
    }


# ============================================================================
# Kernel adapter and TestSpec
# ============================================================================


def numpy_result(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    rotation,
    v_scale,
    mrope_position,
    *,
    head_nums,
    layout_qkv,
    layout_q_out,
    epsilon,
    mrope_section,
    q_quant_mode,
    k_quant_mode,
    q_out_dtype,
):
    """Normalize Kernel/ACLNN/PTA tensors and run the shared NumPy core."""

    return qkv_rms_norm_rope_cache_with_k_scale_numpy(
        to_numpy(qkv),
        to_numpy(q_gamma),
        to_numpy(k_gamma),
        to_numpy(cos_sin),
        to_numpy(slot_mapping),
        to_numpy(k_cache),
        to_numpy(v_cache),
        to_numpy(k_scale_cache),
        to_numpy(query_start_loc),
        to_numpy(seq_lens),
        to_numpy(rotation),
        to_numpy(v_scale),
        to_numpy(mrope_position),
        head_nums=head_nums,
        layout_qkv=layout_qkv,
        layout_q_out=layout_q_out,
        epsilon=epsilon,
        mrope_section=mrope_section,
        q_quant_mode=q_quant_mode,
        k_quant_mode=k_quant_mode,
        q_out_dtype=q_out_dtype,
    )


class QkvRmsNormRopeCacheWithKScaleTestSpec:
    """Kernel-path TestSpec; parameters follow the operator def.cpp."""

    tolerance = TOLERANCE
    compare = staticmethod(compare_outputs)
    customize_inputs = staticmethod(kernel_customize_inputs)

    @staticmethod
    def golden(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc=None,
        seq_lens=None,
        rotation=None,
        v_scale=None,
        mrope_position=None,
        *,
        head_nums,
        layout_qkv="TND",
        layout_q_out="NTD",
        epsilon=1e-6,
        mrope_section=None,
        q_quant_mode="PerTokenPerHead",
        k_quant_mode="PerTokenPerHead",
        q_out_dtype=GE_DT_FLOAT8_E4M3FN,
        **kwargs,
    ):
        result = numpy_result(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            query_start_loc,
            seq_lens,
            rotation,
            v_scale,
            mrope_position,
            head_nums=head_nums,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            epsilon=epsilon,
            mrope_section=mrope_section,
            q_quant_mode=q_quant_mode,
            k_quant_mode=k_quant_mode,
            q_out_dtype=q_out_dtype,
        )
        q_scale = result.q_scale
        if q_scale is None:
            # The raw Kernel ABI always has this physical output.  M-RoPE does
            # not define it, so comparison explicitly ignores this placeholder.
            q_scale = np.zeros(result.q_out.shape[:-1], dtype=np.float32)
        return [
            result.q_out,
            q_scale,
            result.k_cache,
            result.v_cache,
            result.k_scale_cache,
        ]


__spec__ = {
    "qkv_rms_norm_rope_cache_with_k_scale": "QkvRmsNormRopeCacheWithKScaleTestSpec",
}
