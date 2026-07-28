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
import numpy as np

HIF8_POSITIVE_ZERO = 0x00
HIF8_NAN = 0x80
HIF8_POSITIVE_INF = 0x6F
HIF8_NEGATIVE_INF = 0xEF
HIF8_MAX_POSITIVE_NORMAL = 0x6E
HIF8_MAX_NEGATIVE_NORMAL = 0xEE
HIF8_OVERFLOW_THRESHOLD = (2.0**15) * 1.25


def _as_numpy_uint8_codes(codes):
    arr = np.asarray(codes)
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"codes must use an integer dtype, got: {arr.dtype}")
    if arr.size > 0 and arr.dtype != np.uint8:
        min_code = int(arr.min())
        max_code = int(arr.max())
        if min_code < 0 or max_code > 0xFF:
            raise ValueError(
                f"codes must be in [0, 255], got min={min_code}, max={max_code}"
            )
    return arr.astype(np.uint8, copy=False)


def _decode_normal_payload_numpy(payload, d, mantissa_bits):
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa = (payload & mantissa_mask).astype(np.float32, copy=False)
    significand = np.float32(1.0) + mantissa / np.float32(1 << mantissa_bits)
    if d == 0:
        return significand

    exponent_payload = payload >> mantissa_bits
    sign_exp = (exponent_payload >> (d - 1)) & 1
    mag_tail_mask = (1 << (d - 1)) - 1
    magnitude = (1 << (d - 1)) | (exponent_payload & mag_tail_mask)
    exponent = np.where(sign_exp == 0, magnitude, -magnitude).astype(
        np.float32, copy=False
    )
    return np.power(np.float32(2.0), exponent) * significand


def hif8_to_fp32_numpy(codes):
    bits = _as_numpy_uint8_codes(codes).astype(np.int16, copy=False)
    payload = bits & 0x7F
    sign = (bits & 0x80) != 0
    out = np.zeros(bits.shape, dtype=np.float32)

    nan_mask = bits == HIF8_NAN
    if nan_mask.any():
        out[nan_mask] = np.nan

    inf_mask = payload == HIF8_POSITIVE_INF
    if inf_mask.any():
        out[inf_mask] = np.where(sign[inf_mask], -np.inf, np.inf)

    dml_mask = (payload <= 0x07) & ~nan_mask
    if dml_mask.any():
        mantissa = payload[dml_mask]
        nonzero = mantissa != 0
        if nonzero.any():
            values = np.power(
                np.float32(2.0), mantissa[nonzero].astype(np.float32) - np.float32(23.0)
            )
            dml_values = out[dml_mask]
            dml_values[nonzero] = values
            out[dml_mask] = dml_values

    layouts = (
        (0, 0x78, 0x08, 3),
        (1, 0x70, 0x10, 3),
        (2, 0x60, 0x20, 3),
        (3, 0x60, 0x40, 2),
        (4, 0x60, 0x60, 1),
    )

    for d, prefix_mask, prefix_value, mantissa_bits in layouts:
        mask = ((payload & prefix_mask) == prefix_value) & ~inf_mask
        if mask.any():
            out[mask] = _decode_normal_payload_numpy(payload[mask], d, mantissa_bits)

    negative_finite = sign & np.isfinite(out) & (out != 0.0)
    if negative_finite.any():
        out[negative_finite] = -out[negative_finite]
    return out


def _positive_finite_codebook_numpy():
    codes = np.arange(128, dtype=np.uint8)
    values = hif8_to_fp32_numpy(codes)
    finite = np.isfinite(values)
    codes = codes[finite]
    values = values[finite]
    order = np.argsort(values)
    return values[order], codes[order]


def fp32_to_hif8_numpy(values, saturate=False, nan_to_zero=False):
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty(arr.shape, dtype=np.uint8)

    nan_mask = np.isnan(arr)
    if nan_mask.any():
        out[nan_mask] = HIF8_POSITIVE_ZERO if nan_to_zero else HIF8_NAN

    pos_inf_mask = arr == np.inf
    if pos_inf_mask.any():
        out[pos_inf_mask] = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF

    neg_inf_mask = arr == -np.inf
    if neg_inf_mask.any():
        out[neg_inf_mask] = HIF8_MAX_NEGATIVE_NORMAL if saturate else HIF8_NEGATIVE_INF

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return out

    magnitude = np.abs(arr[finite_mask]).astype(np.float32, copy=False)
    pos_values, pos_codes = _positive_finite_codebook_numpy()
    hi = np.searchsorted(pos_values, magnitude, side="left")
    max_idx = pos_values.size - 1
    hi_clamped = np.minimum(hi, max_idx)
    lo_clamped = np.maximum(hi - 1, 0)

    dist_hi = np.abs(pos_values[hi_clamped] - magnitude)
    dist_lo = np.abs(pos_values[lo_clamped] - magnitude)
    rounded_pos = np.where(
        dist_hi <= dist_lo, pos_codes[hi_clamped], pos_codes[lo_clamped]
    ).astype(np.uint8, copy=False)

    overflow_mask = magnitude >= np.float32(HIF8_OVERFLOW_THRESHOLD)
    overflow_code = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF
    rounded_pos = np.where(overflow_mask, np.uint8(overflow_code), rounded_pos).astype(
        np.uint8, copy=False
    )

    negative = arr[finite_mask] < 0
    negative_codes = (rounded_pos.astype(np.int16) | 0x80).astype(np.uint8, copy=False)
    signed_codes = np.where(
        negative & (rounded_pos != 0), negative_codes, rounded_pos
    ).astype(np.uint8, copy=False)
    out[finite_mask] = signed_codes
    return out


# ═══════════════════════════════════════════════════════════════════
# LUT Decode: 256-entry table, pure integer lookup
# ═══════════════════════════════════════════════════════════════════

_LUT_FP32_BITS = None


def _build_lut():
    """Build 256-entry LUT once. Uses baseline decoder — bit-exact by construction."""
    global _LUT_FP32_BITS
    all_codes = np.arange(256, dtype=np.uint8)
    fp32_vals = hif8_to_fp32_numpy(all_codes)
    _LUT_FP32_BITS = fp32_vals.view(np.uint32).copy()


def hif8_to_fp32_lut(codes):
    """Decode HiF8 uint8 -> fp32 via 256-entry LUT.
    Bit-exact equivalent to hif8_to_fp32_numpy, ~24x faster.
    """
    global _LUT_FP32_BITS
    if _LUT_FP32_BITS is None:
        _build_lut()
    flat = codes.ravel().astype(np.uint8)
    bits = _LUT_FP32_BITS[flat]
    return bits.view(np.float32).reshape(codes.shape)


# ═══════════════════════════════════════════════════════════════════
# Valid code computation — for direct uint8 generation (skip encode)
# ═══════════════════════════════════════════════════════════════════
_VALID_CACHE = {}


def get_valid_hif8_codes(min_val, max_val):
    """Return HiF8 codes whose decoded fp32 value is in [min_val, max_val] and finite.
    These are exactly the codes fp32_to_hif8_numpy would produce for uniform input.
    Cached per (min_val, max_val) pair.
    """
    key = (float(min_val), float(max_val))
    if key in _VALID_CACHE:
        return _VALID_CACHE[key].copy()

    all_codes = np.arange(256, dtype=np.uint8)
    all_vals = hif8_to_fp32_numpy(all_codes)
    mask = np.isfinite(all_vals) & (all_vals >= min_val) & (all_vals <= max_val)
    valid = all_codes[mask]

    _VALID_CACHE[key] = valid
    return valid.copy()


_VU_CACHE = {}

# LUT size: 1MB (2^20 entries). Every valid code gets at least 1 entry
# (floor-1 guarantee). Distortion for tiny-probability codes (< 3.8e-7) is
# negligible — they round up from ~0.0004 to 1 entry, < 0.02% total mass shift.
_SAMPLE_LUT_SIZE = 1 << 20


def _build_sampling_lut(codes, vals, min_val, max_val):
    """Build inverse-CDF LUT with floor-1 guarantee for every code."""
    sorted_idx = np.argsort(vals)
    sorted_vals = vals[sorted_idx].astype(np.float64)
    sorted_codes = codes[sorted_idx]
    n = len(sorted_codes)

    mids = (sorted_vals[:-1] + sorted_vals[1:]) * 0.5
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    lo[0] = min_val
    lo[1:] = mids
    hi[:-1] = mids
    hi[-1] = max_val
    widths = np.maximum(hi - lo, 0.0)
    total = widths.sum()

    if total <= 0 or n >= _SAMPLE_LUT_SIZE:
        return sorted_codes  # fallback: uniform over codes

    probs = widths / total
    target = probs * _SAMPLE_LUT_SIZE
    # Floor-1: each code gets at least 1, rest distributed proportionally
    counts = np.ones(n, dtype=np.int32)
    remaining = _SAMPLE_LUT_SIZE - n
    fractional = np.maximum(target - 1.0, 0.0)
    frac_sum = fractional.sum()
    if frac_sum > 0:
        extra = (fractional / frac_sum * remaining).astype(np.int32)
        remainder = fractional / frac_sum * remaining - extra.astype(np.float64)
        extra[np.argsort(-remainder)[: remaining - extra.sum()]] += 1
        counts += extra

    lut = np.repeat(sorted_codes.astype(np.uint8), counts)
    if len(lut) < _SAMPLE_LUT_SIZE:
        lut = np.pad(
            lut, (0, _SAMPLE_LUT_SIZE - len(lut)), constant_values=sorted_codes[-1]
        )
    return lut[:_SAMPLE_LUT_SIZE]


def sample_hif8_value_uniform(min_val, max_val, shape):
    """Generate random HiF8 codes with value-space-uniform distribution.

    256KB LUT with floor-1 guarantee — every valid code appears at least once.
    Tiny probability codes get rounded up (max distortion < 0.02% of total mass).
    """
    key = (float(min_val), float(max_val))
    if key not in _VU_CACHE:
        codes = get_valid_hif8_codes(min_val, max_val)
        vals = hif8_to_fp32_numpy(codes)
        lut = _build_sampling_lut(codes, vals, min_val, max_val)
        _VU_CACHE[key] = lut

    lut = _VU_CACHE[key]
    indices = np.random.randint(0, len(lut), size=shape, dtype=np.int32)
    return lut[indices]
