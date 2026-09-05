#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
indexer_quant_cache kernel (single-op direct-invoke) golden -- SELF-CONTAINED.

Per-block quantize x and scatter into cache[slot] (+ per-block scale). Two in-place outputs
(cache, scale). quant_mode 0=MXFP8 1=Normal 2=HiFloat8(scale suppressed) 3=MXFP4. Mirrors
the kernel math; reuses no external module (per-mode independent-golden convention of this repo).
"""

import numpy as np

__spec__ = {
    "indexer_quant_cache": "IndexerQuantCacheTestSpec",
    "aclnnIndexerQuantCache": "IndexerQuantCacheTestSpec",
}

# Third-party dtype handles are imported lazily (see _ensure_dtypes) so that importing
# the golden_funcs package -- e.g. to run *another* operator's golden -- does not require
# ml_dtypes / en_dtypes to be installed. They are only needed to run this golden.
BF16 = None
F8E4M3 = None
F8E5M2 = None
F8E8M0 = None
FP4E2M1 = None
HAS_FP4 = False
FP4E1M2 = None  # MX-FP4 E1M2 (1 exp + 2 mantissa) fp4 nibble dtype (en_dtypes)
HAS_FP4_E1M2 = False
HIF8 = None
HAS_HIF8 = False
_DTYPES_READY = False


def _ensure_dtypes():
    """Lazily import ml_dtypes (required) + en_dtypes (mode3 only) on first golden call."""
    global BF16, F8E4M3, F8E5M2, F8E8M0, FP4E2M1, HAS_FP4
    global FP4E1M2, HAS_FP4_E1M2, HIF8, HAS_HIF8, _DTYPES_READY
    if _DTYPES_READY:
        return
    try:
        import ml_dtypes
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "ml_dtypes is needed for the indexer_quant_cache golden. "
            "Please install with `pip3 install ml-dtypes`"
        ) from e
    BF16 = ml_dtypes.bfloat16
    F8E4M3 = ml_dtypes.float8_e4m3fn
    F8E5M2 = ml_dtypes.float8_e5m2
    try:
        FP4E2M1 = ml_dtypes.float4_e2m1fn
        HAS_FP4 = True
    except AttributeError:
        HAS_FP4 = False
    try:
        import en_dtypes as _en

        F8E8M0 = _en.float8_e8m0
        FP4E1M2 = _en.float4_e1m2
        HAS_FP4_E1M2 = True
    except (ImportError, AttributeError):
        HAS_FP4_E1M2 = False
    try:
        from en_dtypes import hifloat8 as _HIF8

        HIF8 = _HIF8
        HAS_HIF8 = True
    except ImportError:
        HAS_HIF8 = False
    _DTYPES_READY = True


FP8_E4M3_MAX = np.float32(448.0)
FP8_E5M2_MAX = np.float32(57344.0)
EPS = np.float32(1e-4)
# 1/fp8max float-bit constants used by the Normal-quant kernel (VFProcessDynamicBlockQuant)
INV_FP8_E4M3 = np.uint32(0x3B124925).view(np.float32)
INV_FP8_E5M2 = np.uint32(0x37924925).view(np.float32)

MX_FP8_BLOCK = 32  # MX-FP8 now uses standard 32-element blocks
MX_FP4_BLOCK = 32

# FP16ConvertMXFP4 constants (indexer_quant_cache_mx_fp4_base.h) for the fp16-input MX-FP4 path
SPECIAL_VALUE_E2M1 = 0x00FF  # low-bit mask for E2M1 fp4 half-domain round adjust
SPECIAL_VALUE_E1M2 = 0x007F  # low-bit mask for E1M2
NEW_MANTISSA_MAXFP4 = 0x0008  # guard/sticky bit OR'd before fp16->bf16 truncation


def _to_numpy(t):
    """Materialise a (possibly strided torch) tensor as a contiguous numpy array of its view shape,
    preserving raw bits for sub-byte / fp8 dtypes."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(t, torch.Tensor):
        tc = t.detach().cpu().contiguous()
        if tc.dtype == torch.bfloat16:
            return tc.view(torch.uint16).numpy().view(BF16)
        if str(tc.dtype) in ("torch.float8_e4m3fn",):
            return tc.view(torch.uint8).numpy().view(F8E4M3)
        if str(tc.dtype) in ("torch.float8_e5m2",):
            return tc.view(torch.uint8).numpy().view(F8E5M2)
        if "float8_e8m0" in str(tc.dtype):
            return tc.view(torch.uint8).numpy().view(F8E8M0)
        return tc.numpy()
    return np.ascontiguousarray(t)


def _cache_fp8_type(cache):
    """fp8 element type of the cache tensor (mode 0/1)."""
    try:
        import torch

        if isinstance(cache, torch.Tensor):
            if cache.dtype == torch.float8_e5m2:
                return F8E5M2, FP8_E5M2_MAX, INV_FP8_E5M2
            return F8E4M3, FP8_E4M3_MAX, INV_FP8_E4M3
    except ImportError:
        pass
    dt = str(getattr(cache, "dtype", ""))
    if "e5m2" in dt:
        return F8E5M2, FP8_E5M2_MAX, INV_FP8_E5M2
    return F8E4M3, FP8_E4M3_MAX, INV_FP8_E4M3


def _x_to_f32(x):
    """Load x exactly like the kernel: bf16/fp16 -> fp32."""
    return _to_numpy(x).astype(np.float32)


# ---------------- quant_mode 0 : MX-FP8 (block 32, e8m0 scale) ----------------
def _round_scale_pow2_kernel(s):
    """Replicate VFProcessDynamicMxFp8Quant roundScale (== swiglu mxfp8): correct mantissa
    check -> e8m0 = exp + (mantissa != 0), divisor = 2^(e8m0 - 127)."""
    bits = np.float32(s).view(np.uint32)
    exp = np.int64((bits >> np.uint32(23)) & np.uint32(0xFF))
    man = np.int64(bits & np.uint32(0x7FFFFF))
    exp_scale = exp - 127 + (1 if man != 0 else 0)
    e8m0 = np.uint8((exp_scale + 127) & 0xFF)
    s_div = np.float32(np.uint32((exp_scale + 127) << 23).view(np.float32))
    return s_div, e8m0


def _encode_mxfp8_row(x_f32_row, d, fp8_type, fp8_max, round_scale):
    # MX-FP8 的 scale 恒为 e8m0 (2 的幂), kernel 只实现 round 路径, 故 golden 也恒 round (忽略 round_scale)。
    del round_scale
    scale_col = (d + MX_FP8_BLOCK - 1) // MX_FP8_BLOCK
    coeff = np.float32(1.0) / np.float32(fp8_max)
    cache = np.zeros(d, dtype=np.uint8)
    e8m0 = np.zeros(scale_col, dtype=np.uint8)
    for g in range(scale_col):
        blk = x_f32_row[g * MX_FP8_BLOCK : (g + 1) * MX_FP8_BLOCK]
        m = np.float32(np.max(np.abs(blk)))
        m = np.maximum(m, EPS)
        s = np.float32(m * coeff)
        s_div, e = _round_scale_pow2_kernel(s)
        q = blk / s_div
        q = np.minimum(np.maximum(q, -fp8_max), fp8_max)
        cache[g * MX_FP8_BLOCK : g * MX_FP8_BLOCK + blk.shape[0]] = q.astype(
            fp8_type
        ).view(np.uint8)
        e8m0[g] = e
    return cache.view(fp8_type), e8m0


# ---------------- quant_mode 1 : Normal whole-row quant (one float32 scale per row) ----------------
def _encode_normal_row(x_f32_row, d, fp8_type, fp8_max, inv_fp8max, round_scale):
    # round_scale=1 rounds the row scale up to the nearest power of two, mirroring the kernel
    # VFProcessDynamicBlockQuant roundScale branch (== kv_compress_epilog roundScale); the rounded
    # value is both the divisor and the stored float32 scale.
    cache = np.zeros(d, dtype=np.uint8)
    m = np.float32(np.max(np.abs(x_f32_row)))
    if m != np.float32(0.0):
        s = np.float32(m * inv_fp8max)  # rowmax / fp8max
        if round_scale:
            s, _ = _round_scale_pow2_kernel(s)
        q = x_f32_row / s
    else:
        s = np.float32(0.0)
        q = x_f32_row
    cache[:] = q.astype(fp8_type).view(np.uint8)
    # Degenerate rowmax: for x = ±inf the rowmax is inf and the float32 scale overflows
    # to inf in this reference, but the device emits 0.0 for a non-finite rowmax (the
    # quantized fp8 cache itself still matches under requant). Replicate the device's
    # stored scale so the float32 scale output compares exactly.
    s_store = np.float32(0.0) if not np.isfinite(m) else s
    return cache.view(fp8_type), np.array([s_store], dtype=np.float32)  # scaleCol == 1


# ---------------- quant_mode 2 : HiFloat8 (cache only; kernel does not write scale) ----------------
HIFLOAT8_MAX_VALUE = np.float32(32768.0)


def _encode_hifloat8_row(x_f32_row, d, scale_attr):
    """Mirror VFProcessHifp8Quant: y = x * scale_attr, cast to hifloat8 (round-nearest).
    The kernel writes hifloat8 bytes into the cache buffer and does NOT write the scale
    output, so only the cache (output 0) is verified for mode 2.

    Device hifloat8 cast (SatMode::SAT) is a FINITE format: it saturates overflow/±inf
    to the max finite magnitude and maps NaN -> 0 (0x00), whereas en_dtypes emits the
    inf/nan codes. Replicate the device clamp so NaN/inf inputs compare byte-exact."""
    if not HAS_HIF8:
        raise RuntimeError("HiFloat8 golden needs en_dtypes.hifloat8")
    y = x_f32_row * np.float32(scale_attr)
    y = np.where(np.isnan(y), np.float32(0.0), y)
    y = np.clip(y, np.float32(-HIFLOAT8_MAX_VALUE), np.float32(HIFLOAT8_MAX_VALUE))
    y = y.astype(HIF8).view(np.uint8)
    return y, np.zeros((d + MX_FP8_BLOCK - 1) // MX_FP8_BLOCK, dtype=np.float32)


# ---------------- quant_mode 3 : MX-FP4 (block 32, e8m0 scale, packed) ----------------
BF16_EXP_INF_NAN = np.int32(
    0xFF
)  # bf16 8-bit exp field value for INF/NaN (kernel: BF16_EMASK_AND_INF_VAL_MAXFP4=0x7f80 -> exp field = 0xFF)
FP8_E8M0_NAN_VAL = np.uint8(0xFF)  # e8m0 NaN code (kernel: FP8_E8M0_NAN_VAL_MAXFP4)


def _mxfp4_block_nan_fixup(e_block, e8m0, inv_scale):
    """Mirror the device NaN/INF special-case in vfComputeScaleMXFP4:
      - maxExp exp field == 0xFF (bf16 INF/NaN) -> e8m0 = 0xFF (NaN code), halfScale = bf16 NaN
      - maxExp == 0 (all-zero block)            -> e8m0 = 0,    halfScale = 0
    Returns the corrected e8m0 and a mask of blocks whose halfScale is NaN."""
    is_zero = e_block == 0
    is_inf_nan = e_block == BF16_EXP_INF_NAN
    e8m0 = np.where(is_inf_nan, FP8_E8M0_NAN_VAL.astype(np.uint8), e8m0)
    nan_scale_mask = is_inf_nan
    inv_scale = np.where(is_zero, np.float32(0.0), inv_scale)
    inv_scale = np.where(nan_scale_mask, np.float32(np.nan), inv_scale)
    return e8m0, nan_scale_mask, inv_scale


def _mxfp4_cast_data(y_scaled, fp4_dt, nan_scale_mask, n_scale, d):
    """Cast scaled bf16 data to fp4 nibbles, applying the device NaN->+0 rule.
    The device cast (CAST_RINT) maps bf16 NaN to fp4 +0 (0x0), whereas en_dtypes
    emits fp4 -0 (0x8). Force NaN-amax blocks to +0 to match the device."""
    nibble = y_scaled.astype(fp4_dt).view(np.uint8).reshape(d)
    if nan_scale_mask.any():
        nibble = nibble.copy()
        for g in range(n_scale):
            if nan_scale_mask[g]:
                nibble[g * MX_FP4_BLOCK : (g + 1) * MX_FP4_BLOCK] = 0
    return nibble


def _encode_mxfp4_row(x_bf16_row, d, is_e1m2=False):
    """Mirror VFProcessDynamicMxFp4Quant: per-32-block shared exponent taken element-first
    (bf16 exponent field max-reduced over the block), then e8m0 = max(E_block - f4Emax_exp, 0),
    inv_scale = 2^(127 - e8m0), y_scaled(bf16) -> fp4 (e2m1 or e1m2), packed 2/byte.
    f4Emax_exp = 2 for E2M1 (FP4_E2M1_BF16_MAX_EXP=0x0100) and 0 for E1M2 (FP4_E1M2_MAX_EXP=0x0000).

    NaN/INF handling mirrors vfComputeScaleMXFP4: when the block maxExp is 0x7f80 (bf16 INF/NaN
    exp field) the device stores e8m0=0xFF and a bf16 NaN halfScale, so y_scaled = x * NaN = NaN
    and the fp4 cast (CAST_RINT) emits +0 (0x0)."""
    if is_e1m2:
        if not HAS_FP4_E1M2:
            raise RuntimeError("MX-FP4 E1M2 golden needs en_dtypes.float4_e1m2")
        fp4_dt = FP4E1M2
        f4emax_exp = 0
    else:
        if not HAS_FP4:
            raise RuntimeError("MX-FP4 golden needs ml_dtypes.float4_e2m1fn")
        fp4_dt = FP4E2M1
        f4emax_exp = 2
    n_scale = (d + MX_FP4_BLOCK - 1) // MX_FP4_BLOCK
    y_bf16 = x_bf16_row.astype(BF16)
    bits = y_bf16.view(np.uint16)
    e_field = ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(
        np.int32
    )  # per-element biased bf16 exp
    blk = e_field.reshape(n_scale, MX_FP4_BLOCK)
    e_block = blk.max(axis=-1)  # element-first max exponent
    e8m0 = np.maximum(e_block - f4emax_exp, 0).astype(
        np.uint8
    )  # f4Emax exp: e2m1=2, e1m2=0
    inv_scale = np.exp2(
        np.float32(127.0) - e8m0.astype(np.float32)
    )  # 2^(127-e8m0), exact pow2
    e8m0, nan_scale_mask, inv_scale = _mxfp4_block_nan_fixup(e_block, e8m0, inv_scale)
    y = y_bf16.astype(np.float32).reshape(n_scale, MX_FP4_BLOCK)
    y_scaled = (
        (y * inv_scale[:, None]).astype(BF16).astype(np.float32)
    )  # bf16 mul (exact: pow2)
    nibble = _mxfp4_cast_data(y_scaled, fp4_dt, nan_scale_mask, n_scale, d)
    return nibble, e8m0


def _encode_mxfp4_row_fp16(x_fp16_row, d, is_e1m2=False):
    """fp16-input MX-FP4 golden mirroring the DEVICE half datapath exactly
    (indexer_quant_cache_mx_fp4_base.h: vfComputeMaxExpMXFP4 + vfComputeScaleMXFP4 +
    vfComputeDataMXFP4 with FP16ConvertMXFP4). Unlike the bf16 path, the device does NOT
    downcast fp16->bf16 before deciding the fp4 rounding: it first applies FP16ConvertMXFP4
    (a half-domain bit manipulation that ORs NEW_MANTISSA=0x0008 into values whose low
    mantissa bits are nonzero-but-<0x0008) on the fp16 bits, THEN truncates fp16->bf16
    (CAST_TRUNC), multiplies by the pow2 halfScale in bf16, and casts bf16->fp4 (CAST_RINT).

    NaN/INF handling: same as the bf16 path -- when block maxExp == 0x7f80 the device
    stores e8m0=0xFF and a bf16 NaN halfScale, so y_scaled = NaN and fp4 cast -> +0 (0x0)."""
    if is_e1m2:
        if not HAS_FP4_E1M2:
            raise RuntimeError("MX-FP4 E1M2 golden needs en_dtypes.float4_e1m2")
        fp4_dt = FP4E1M2
        f4emax_exp = 0
        special_value = np.uint16(SPECIAL_VALUE_E1M2)  # 0x007f
    else:
        if not HAS_FP4:
            raise RuntimeError("MX-FP4 golden needs ml_dtypes.float4_e2m1fn")
        fp4_dt = FP4E2M1
        f4emax_exp = 2
        special_value = np.uint16(SPECIAL_VALUE_E2M1)  # 0x00ff
    n_scale = (d + MX_FP4_BLOCK - 1) // MX_FP4_BLOCK

    x16 = np.ascontiguousarray(x_fp16_row).astype(np.float16)
    bits16 = x16.view(np.uint16)

    # ---- exponent / e8m0 / inv_scale: raw fp16 -> bf16 (CAST_TRUNC) -> bf16 exp field ----
    f32 = x16.astype(np.float32)
    bf16_trunc_bits = (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    e_field = ((bf16_trunc_bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int32)
    blk = e_field.reshape(n_scale, MX_FP4_BLOCK)
    e_block = blk.max(axis=-1)  # element-first max exponent
    e8m0 = np.maximum(e_block - f4emax_exp, 0).astype(
        np.uint8
    )  # f4Emax exp: e2m1=2, e1m2=0
    inv_scale = np.exp2(
        np.float32(127.0) - e8m0.astype(np.float32)
    )  # halfScale = 2^(127-e8m0)
    e8m0, nan_scale_mask, inv_scale = _mxfp4_block_nan_fixup(e_block, e8m0, inv_scale)

    # ---- FP16ConvertMXFP4 half-domain bit manipulation (on the fp16 bits) ----
    and_result = bits16 & special_value
    special_mask = (and_result > np.uint16(0)) & (
        and_result < np.uint16(NEW_MANTISSA_MAXFP4)
    )
    new_value = bits16 | np.uint16(NEW_MANTISSA_MAXFP4)  # OR bit 3 (0x0008)
    adj_bits = np.where(special_mask, new_value, bits16).astype(np.uint16)
    adj_fp16 = adj_bits.view(np.float16)

    # ---- adjusted fp16 -> bf16 (CAST_TRUNC), then bf16 mul by pow2 halfScale, then fp4 RINT ----
    adj_f32 = adj_fp16.astype(np.float32)
    adj_bf16_bits = (adj_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    y_bf16 = adj_bf16_bits.view(BF16)
    y = y_bf16.astype(np.float32).reshape(n_scale, MX_FP4_BLOCK)
    y_scaled = (
        (y * inv_scale[:, None]).astype(BF16).astype(np.float32)
    )  # bf16 mul (exact: pow2)
    nibble = _mxfp4_cast_data(y_scaled, fp4_dt, nan_scale_mask, n_scale, d)
    return nibble, e8m0


def _indexer_core(
    indexer_compress_cache,
    indexer_compress_cache_scale,
    x,
    slot_mapping,
    quant_mode=1,
    round_scale=True,
    scale=1.0,
):
    """Compute the in-place cache + scale in flattened logical VIEW shape.

    Non-contiguity of the 4D paged [blockNum, blockSize, 1, headDim] layout is handled by
    TTK's as_strided readback from the CSV storage/stride; the golden indexes the flattened
    [numSlots, col] view directly (slot == block*blockSize + pos)."""
    quant_mode = int(quant_mode)
    round_scale = int(round_scale)

    cache = _to_numpy(indexer_compress_cache).copy()  # view shape, fp8/uint8
    scale_arr = _to_numpy(
        indexer_compress_cache_scale
    ).copy()  # view shape, e8m0/float32
    cache2d = cache.reshape(-1, cache.shape[-1])
    scale2d = scale_arr.reshape(-1, scale_arr.shape[-1])

    x_f32 = _x_to_f32(x)
    x_bf16 = _to_numpy(x)
    # The op flattens ALL leading dims of x -> bs (= prod(shape[:-1])); d = last dim.
    x_f32 = x_f32.reshape(-1, x_f32.shape[-1])  # [bs, d]
    x_bf16 = x_bf16.reshape(-1, x_bf16.shape[-1])
    bs, d = x_f32.shape
    # slot_mapping may be 1D [bs] or N-D (e.g. 2D [bs,1] for 3D x); flatten to bs.
    sm = _to_numpy(slot_mapping).reshape(-1).astype(np.int64)

    if quant_mode in (0, 1):
        fp8_type, fp8_max, inv_fp8max = _cache_fp8_type(indexer_compress_cache)
    # MX-FP4 sub-format (E2M1 vs E1M2) is determined by the cache output dtype, not an attr.
    cache_is_e1m2 = "e1m2" in str(cache2d.dtype).lower()

    for i in range(bs):
        slot = int(sm[i])
        if slot < 0:
            continue
        if quant_mode == 0:
            c, s = _encode_mxfp8_row(x_f32[i], d, fp8_type, fp8_max, round_scale)
        elif quant_mode == 1:
            c, s = _encode_normal_row(
                x_f32[i], d, fp8_type, fp8_max, inv_fp8max, round_scale
            )
        elif quant_mode == 2:
            c, s = _encode_hifloat8_row(
                x_f32[i], d, scale
            )  # scale output unused (s ignored)
        elif quant_mode == 3:
            if x_bf16.dtype == BF16:
                c, s = _encode_mxfp4_row(x_bf16[i], d, is_e1m2=cache_is_e1m2)
            else:
                # fp16 input: device keeps fp16 precision through FP16ConvertMXFP4 before the
                # bf16 truncation/quant -> use the half-domain datapath, NOT a bf16 downcast.
                c, s = _encode_mxfp4_row_fp16(
                    x_f32[i].astype(np.float16), d, is_e1m2=cache_is_e1m2
                )
        else:
            raise NotImplementedError(
                "indexer_quant_cache golden: quant_mode should be in [0,3], got %d"
                % quant_mode
            )
        # cache: same itemsize as cache2d -> bit-reinterpret。只写前 len(c) 列, cache 行更宽时余下保持原值。
        cv = c.view(cache2d.dtype)
        cache2d[slot, : cv.shape[0]] = cv
        if quant_mode == 2:
            continue  # mode 2 (HiFloat8): kernel does not write the scale output -> leave it suppressed
        # scale: e8m0 is a raw 1-byte code (bit-reinterpret, NOT numeric astype); float32 is a value
        if s.dtype == scale2d.dtype:
            sv = s
        elif s.itemsize == scale2d.dtype.itemsize:
            sv = s.view(scale2d.dtype)
        else:
            sv = s.astype(scale2d.dtype)
        scale2d[slot, : sv.shape[0]] = sv

    # mode 2 scale is never written by the kernel -> None (TTK marks the output SUPPRESSED)
    scale_ret = None if quant_mode == 2 else scale_arr
    return cache, scale_ret


class IndexerQuantCacheTestSpec:
    """TTK 3.0 TestSpec for the kernel-level indexer_quant_cache operator."""

    @staticmethod
    def golden(
        cache,
        cache_scale,
        x,
        slot_mapping,
        quant_mode=1,
        round_scale=True,
        x_scale=1.0,
        **kwargs,
    ):
        # Kernel/GE pass snake_case attrs, while ACLNN follows the C API names.
        quant_mode = kwargs.pop("quantMode", quant_mode)
        round_scale = kwargs.pop("roundScale", round_scale)
        x_scale = kwargs.pop("xScale", x_scale)
        _ensure_dtypes()
        cache_out, scale_out = _indexer_core(
            cache,
            cache_scale,
            x,
            slot_mapping,
            quant_mode=int(quant_mode),
            round_scale=int(round_scale),
            scale=float(x_scale),
        )
        if scale_out is None:
            return [np.ascontiguousarray(cache_out), None]
        return [np.ascontiguousarray(cache_out), np.ascontiguousarray(scale_out)]

    @staticmethod
    def customize_inputs(
        cache,
        cache_scale,
        x,
        slot_mapping,
        quant_mode=1,
        round_scale=True,
        x_scale=1.0,
        **kwargs,
    ):
        """Keep CSV-generated data, but make slot indices legal, unique and well spread."""
        bs = int(np.prod(tuple(x.shape[:-1])))
        num_slots = int(cache.shape[0] * cache.shape[1])
        if bs > num_slots:
            raise ValueError(
                f"bs ({bs}) must not exceed numSlots ({num_slots}) in this TTK input"
            )

        if hasattr(cache, "copy_"):
            # ACLNN path: TTK passes torch tensors and consumes mutations in place.
            import torch

            slots = torch.arange(bs, dtype=torch.int64, device=slot_mapping.device)
            slots = (slots * num_slots // bs).to(dtype=slot_mapping.dtype)
            if bs > 1:
                slots[-1] = -1
            slot_mapping.copy_(slots.reshape(slot_mapping.shape))
            return (cache, cache_scale, x, slot_mapping)

        slots = np.arange(bs, dtype=np.int64) * num_slots // bs
        slots = slots.astype(slot_mapping.dtype)
        if bs > 1:
            slots[
                -1
            ] = -1  # cover the documented skip path without introducing duplicate slots
        slot_data = slots.reshape(slot_mapping.shape)
        slot_mapping[...] = slot_data
        return (cache, cache_scale, x, slot_data)

    tolerance = {
        "float4_e2m1": {"standard": "binary_equal"},
        "float4_e1m2": {"standard": "binary_equal"},
        "float8_e8m0": {"standard": "binary_equal"},
    }
