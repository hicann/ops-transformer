#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
    "kernel": {
        "grouped_matmul_activation_quant": "grouped_matmul_activation_quant_golden"
    }
}

import numpy as np
import torch

try:
    import ml_dtypes
except ModuleNotFoundError:
    ml_dtypes = None


MXFP_GROUP_SIZE = 64
MXFP_SCALE_PAIR = 2
FLOAT_EPS = 1e-6
MXFP_SCALE_ELEMENT_NUM = 32
N_ALIGN_SIZE = 64
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FP4_E2M1_MAX = 6.0
FP4_E1M2_MAX = 3.5
QUANT_DTYPE_E5M2 = 35
QUANT_DTYPE_E4M3FN = 36
QUANT_DTYPE_E2M1 = 40
QUANT_DTYPE_E1M2 = 41
SCALE_ALG_OCP = 0
SCALE_ALG_DYNAMIC_DTYPE_RANGE = 2
BF16_EXP_MASK = 0x7F80
BF16_EXP_BIAS = 0x7F00
FP8_E4M3_MAX_EXP = 0x0400
FP8_E5M2_MAX_EXP = 0x0780
FP4_E2M1_MAX_EXP = 0x0100
FP4_E1M2_MAX_EXP = 0x0000
MAX_EXP_FOR_FP8 = 0x00FF
NAN_CUSTOMIZATION = 0x7F81
SPECIAL_EXP_THRESHOLD = 0x0040
GELU_TANH_NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715
GELU_TANH_APPROX_FACTOR = 1.0 / 0.044715


def grouped_matmul_activation_quant_golden(
    x,
    group_list,
    weight,
    weight_scale,
    bias,
    activation_type: str,
    *,
    x_scale=None,
    transpose_weight: bool = False,
    group_list_type: int = 0,
    tuning_config=None,
    quant_mode: str = "",
    y_dtype: int = QUANT_DTYPE_E4M3FN,
    round_mode: str = "rint",
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
    **kwargs,
):
    del bias, tuning_config, round_mode
    if activation_type != "gelu_tanh":
        raise ValueError("activation_type only supports gelu_tanh.")
    if quant_mode in (None, ""):
        quant_mode = "mx"
    if quant_mode != "mx":
        raise ValueError("quant_mode 参数当前仅支持为 mx")
    if x_scale is None:
        raise ValueError("x_scale is required for mx quant mode.")

    output_dtypes = kwargs.get("output_dtypes", [])
    y_dtype = output_dtypes[0] if output_dtypes else _ge_dtype_to_name(y_dtype)
    y_scale_dtype = output_dtypes[1] if len(output_dtypes) > 1 else "float8_e8m0"
    y_dtype = _normalize_dtype_name(y_dtype)
    y_scale_dtype = _normalize_dtype_name(y_scale_dtype)

    x_fp32 = _to_float32(x)
    x_scale_fp32 = _to_float32(x_scale)
    weight_fp32 = _to_float32(_first_tensor(weight))
    weight_scale_fp32 = _to_float32(_first_tensor(weight_scale))
    group_list_cumsum = _get_group_list_cumsum(group_list, group_list_type)

    m, k = x_fp32.shape
    n = _get_n_size(weight_fp32, weight_scale_fp32, transpose_weight)
    if n % N_ALIGN_SIZE != 0:
        raise ValueError(
            "N must be a multiple of {}, actual is {}.".format(N_ALIGN_SIZE, n)
        )

    x_dequant = _dequant_x(x_fp32, x_scale_fp32)
    mm_out = np.zeros((m, n), dtype=np.float32)
    pre_m = 0
    for group_idx, cur_m in enumerate(group_list_cumsum):
        cur_m = int(cur_m)
        if cur_m <= pre_m:
            pre_m = cur_m
            continue
        weight_dequant = _dequant_weight(
            weight_fp32, weight_scale_fp32, group_idx, k, n, transpose_weight
        )
        mm_out[pre_m:cur_m, :] = _torch_matmul(
            x_dequant[pre_m:cur_m, :], weight_dequant
        )
        pre_m = cur_m

    activated = _to_bfloat16_float32(_gelu_tanh(mm_out))
    if scale_alg == SCALE_ALG_OCP:
        y_fp32, y_scale = _dynamic_quant_mx_ocp(activated, y_dtype)
    elif scale_alg == SCALE_ALG_DYNAMIC_DTYPE_RANGE:
        y_fp32, y_scale = _dynamic_quant_mx_dynamic_dtype_range(activated, dst_type_max)
    else:
        y_fp32, y_scale = _dynamic_quant_mx(activated, y_dtype)
    return _cast_mx(y_fp32, y_dtype), _astype(y_scale, y_scale_dtype)


def _get_group_list_cumsum(group_list, group_list_type):
    group_list = np.array(group_list, dtype=np.int64)
    return np.cumsum(group_list) if group_list_type == 1 else group_list


def _to_float32(x):
    return np.asarray(x).astype(np.float32)


def _first_tensor(x):
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    return x


def _dequant_x(x, x_scale):
    m, k = x.shape
    x_scale_flat = x_scale.reshape(m, -1)
    x_scale_broadcast = np.repeat(x_scale_flat, MXFP_SCALE_ELEMENT_NUM, axis=-1)
    k_aligned = x_scale_broadcast.shape[-1]
    x_padded = np.zeros((m, k_aligned), dtype=np.float32)
    x_padded[:, :k] = x
    return x_padded * x_scale_broadcast


def _dequant_weight(weight, weight_scale, group_idx, k, n, transpose_weight):
    weight_g = _get_weight_group(weight, group_idx, k, n, transpose_weight)
    weight_scale_g = _get_weight_scale_group(weight_scale, group_idx, transpose_weight)
    weight_scale_broadcast = np.repeat(weight_scale_g, MXFP_SCALE_ELEMENT_NUM, axis=0)
    k_aligned = weight_scale_broadcast.shape[0]
    weight_padded = np.zeros((k_aligned, n), dtype=np.float32)
    weight_padded[: weight_g.shape[0], : weight_g.shape[1]] = weight_g[:, :n]
    return weight_padded * weight_scale_broadcast


def _get_weight_group(weight, group_idx, k, n, transpose_weight):
    weight_g = weight[group_idx]
    if weight_g.ndim == 2:
        return weight_g.T if transpose_weight else weight_g

    if weight_g.ndim != 4:
        raise ValueError("weight must be logical 3D or FRACTAL_NZ storage 5D.")

    if transpose_weight:
        k1, n1, n0, k0 = weight_g.shape
        weight_kn = weight_g.transpose(0, 3, 1, 2).reshape(k1 * k0, n1 * n0)
    else:
        n1, k1, k0, n0 = weight_g.shape
        weight_kn = weight_g.transpose(1, 2, 0, 3).reshape(k1 * k0, n1 * n0)
    return weight_kn[:k, :n]


def _get_weight_scale_group(weight_scale, group_idx, transpose_weight):
    scale_g = weight_scale[group_idx]
    if transpose_weight:
        n, k_blocks, pair = scale_g.shape
        return scale_g.reshape(n, k_blocks * pair).T
    k_blocks, n, pair = scale_g.shape
    return scale_g.transpose(0, 2, 1).reshape(k_blocks * pair, n)


def _get_n_size(weight, weight_scale, transpose_weight):
    if weight_scale.ndim >= 4:
        return weight_scale.shape[1] if transpose_weight else weight_scale.shape[2]
    if weight.ndim >= 5:
        return (
            weight.shape[2] * weight.shape[3]
            if transpose_weight
            else weight.shape[1] * weight.shape[4]
        )
    if weight.ndim >= 3:
        return weight.shape[-2] if transpose_weight else weight.shape[-1]
    raise ValueError("can not infer N from weight or weight_scale")


def _gelu_tanh(x):
    x = x.astype(np.float32)
    x_torch = torch.from_numpy(x)
    act = x_torch * x_torch
    act = act * x_torch
    act = act + x_torch * np.float32(GELU_TANH_APPROX_FACTOR)
    act = act * np.float32(GELU_TANH_NEG_SQRT_EIGHT_OVER_PI)
    act = torch.exp(act)
    act = act + np.float32(1.0)
    return (x_torch / act).numpy()


def _torch_matmul(x, weight):
    x_torch = torch.from_numpy(x.astype(np.float32))
    weight_torch = torch.from_numpy(weight.astype(np.float32))
    return torch.matmul(x_torch, weight_torch).numpy()


def _to_bfloat16_float32(x):
    return (
        torch.from_numpy(x.astype(np.float32))
        .to(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def _dynamic_quant_mx(x, y_dtype, dst_type_max=0.0):
    dtype_max = dst_type_max if dst_type_max != 0.0 else _get_dtype_max(y_dtype)
    m, n = x.shape
    scale_block_n = int(np.ceil(n / MXFP_GROUP_SIZE))
    scale_n = scale_block_n * MXFP_SCALE_PAIR
    y = np.zeros_like(x, dtype=np.float32)
    y_scale = np.ones((m, scale_n), dtype=np.float32)
    for row_idx in range(m):
        for scale_idx in range(scale_n):
            col_start = scale_idx * MXFP_SCALE_ELEMENT_NUM
            col_end = col_start + MXFP_SCALE_ELEMENT_NUM
            block = x[row_idx, col_start:col_end]
            max_abs = np.max(np.abs(block)) if block.size > 0 else 0.0
            if max_abs == 0.0:
                y_scale[row_idx, scale_idx] = _e8m0_byte_to_float(0)
                continue
            scale = max_abs / dtype_max
            scale = _ceil_power_of_two(scale)
            y_scale[row_idx, scale_idx] = scale
            y[row_idx, col_start:col_end] = block / scale
    return y, y_scale.reshape(m, scale_block_n, MXFP_SCALE_PAIR)


def _dynamic_quant_mx_ocp(x, y_dtype):
    dtype_max_exp = _get_dtype_max_exp(y_dtype)
    m, n = x.shape
    scale_block_n = int(np.ceil(n / MXFP_GROUP_SIZE))
    scale_n = scale_block_n * MXFP_SCALE_PAIR
    y = np.zeros_like(x, dtype=np.float32)
    y_scale = np.zeros((m, scale_n), dtype=np.float32)
    for row_idx in range(m):
        for scale_idx in range(scale_n):
            col_start = scale_idx * MXFP_SCALE_ELEMENT_NUM
            col_end = min(col_start + MXFP_SCALE_ELEMENT_NUM, n)
            block = x[row_idx, col_start:col_end]
            if block.size == 0:
                continue
            max_exp = int(np.max(_bf16_exp_bits(block)))
            scale_byte, half_scale = _compute_ocp_scale(max_exp, dtype_max_exp)
            y_scale[row_idx, scale_idx] = _e8m0_byte_to_float(scale_byte)
            y[row_idx, col_start:col_end] = _bf16_mul_float32(block, half_scale)
    return y, y_scale.reshape(m, scale_block_n, MXFP_SCALE_PAIR)


def _dynamic_quant_mx_dynamic_dtype_range(x, dst_type_max):
    """scaleAlg=2 for FLOAT4_E2M1."""
    if abs(dst_type_max) <= FLOAT_EPS:
        dst_type_max = 0.0
    if dst_type_max not in (0.0, 6.0, 7.0):
        return _dynamic_quant_mx(x, "float4_e2m1", dst_type_max)

    add_value = 0x001F if dst_type_max == 7.0 else 0x003F
    m, n = x.shape
    scale_block_n = int(np.ceil(n / MXFP_GROUP_SIZE))
    scale_n = scale_block_n * MXFP_SCALE_PAIR
    y = np.zeros_like(x, dtype=np.float32)
    y_scale = np.zeros((m, scale_n), dtype=np.float32)
    for row_idx in range(m):
        for scale_idx in range(scale_n):
            col_start = scale_idx * MXFP_SCALE_ELEMENT_NUM
            col_end = min(col_start + MXFP_SCALE_ELEMENT_NUM, n)
            block = x[row_idx, col_start:col_end]
            if block.size == 0:
                continue
            bf16_bits = np.asarray(block, dtype=np.float32).view(np.uint32) >> 16
            max_value = int(np.max(bf16_bits & 0x7FFF))
            max_exp_only = max_value & BF16_EXP_MASK
            if max_exp_only == BF16_EXP_MASK:
                scale_byte = MAX_EXP_FOR_FP8
                half_scale = _bf16_bits_to_float32(NAN_CUSTOMIZATION)
            else:
                rounded_max_exp = (max_value + add_value) & BF16_EXP_MASK
                rounded_max_exp = max(rounded_max_exp, FP4_E2M1_MAX_EXP)
                shared_exp = (rounded_max_exp - FP4_E2M1_MAX_EXP) & 0xFFFF
                scale_byte = (shared_exp >> 7) & 0xFF
                if shared_exp == 0:
                    half_scale_bits = 0
                elif shared_exp == BF16_EXP_BIAS:
                    half_scale_bits = SPECIAL_EXP_THRESHOLD
                else:
                    half_scale_bits = (BF16_EXP_BIAS - shared_exp) & 0xFFFF
                half_scale = _bf16_bits_to_float32(half_scale_bits)
            y_scale[row_idx, scale_idx] = _e8m0_byte_to_float(scale_byte)
            y[row_idx, col_start:col_end] = _bf16_mul_float32(block, half_scale)
    return y, y_scale.reshape(m, scale_block_n, MXFP_SCALE_PAIR)


def _bf16_exp_bits(x):
    x_f32 = np.asarray(x, dtype=np.float32)
    bf16_bits = x_f32.view(np.uint32) >> 16
    return (bf16_bits & BF16_EXP_MASK).astype(np.uint16)


def _compute_ocp_scale(max_exp, fp8_max_exp):
    if max_exp <= fp8_max_exp:
        max_exp = fp8_max_exp
    shared_exp = (max_exp - fp8_max_exp) & 0xFFFF
    if max_exp == BF16_EXP_MASK:
        scale_byte = MAX_EXP_FOR_FP8
        half_scale_bits = NAN_CUSTOMIZATION
    else:
        scale_byte = (shared_exp >> 7) & 0xFF
        if shared_exp == 0:
            half_scale_bits = 0
        elif shared_exp == BF16_EXP_BIAS:
            half_scale_bits = SPECIAL_EXP_THRESHOLD
        else:
            half_scale_bits = (BF16_EXP_BIAS - shared_exp) & 0xFFFF
    return scale_byte, _bf16_bits_to_float32(half_scale_bits)


def _bf16_bits_to_float32(bits):
    return np.array([np.uint32(bits) << 16], dtype=np.uint32).view(np.float32)[0]


def _e8m0_byte_to_float(scale_byte):
    if scale_byte == MAX_EXP_FOR_FP8:
        return np.nan
    return float(np.exp2(np.int32(scale_byte) - 127))


def _bf16_mul_float32(x, value):
    x_torch = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16)
    scale_torch = torch.tensor(value, dtype=torch.float32).to(torch.bfloat16)
    return (x_torch * scale_torch).to(torch.float32).numpy()


def _ceil_power_of_two(value):
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(np.exp2(np.ceil(np.log2(value))))


def _ge_dtype_to_name(dtype):
    dtype_names = {
        QUANT_DTYPE_E5M2: "float8_e5m2",
        QUANT_DTYPE_E4M3FN: "float8_e4m3fn",
        QUANT_DTYPE_E2M1: "float4_e2m1",
        QUANT_DTYPE_E1M2: "float4_e1m2",
    }
    if dtype not in dtype_names:
        raise ValueError("unsupported y_dtype: {}".format(dtype))
    return dtype_names[dtype]


def _get_dtype_max(dtype_name):
    dtype_max = {
        "float8_e5m2": FP8_E5M2_MAX,
        "float8_e4m3fn": FP8_E4M3_MAX,
        "float4_e2m1": FP4_E2M1_MAX,
        "float4_e1m2": FP4_E1M2_MAX,
    }
    if dtype_name not in dtype_max:
        raise ValueError("unsupported y dtype: {}".format(dtype_name))
    return dtype_max[dtype_name]


def _get_dtype_max_exp(dtype_name):
    dtype_max_exp = {
        "float8_e5m2": FP8_E5M2_MAX_EXP,
        "float8_e4m3fn": FP8_E4M3_MAX_EXP,
        "float4_e2m1": FP4_E2M1_MAX_EXP,
        "float4_e1m2": FP4_E1M2_MAX_EXP,
    }
    if dtype_name not in dtype_max_exp:
        raise ValueError("unsupported y dtype: {}".format(dtype_name))
    return dtype_max_exp[dtype_name]


def _cast_mx(x, dtype_name):
    dtype_name = _normalize_dtype_name(dtype_name)
    if dtype_name in ("float4_e2m1", "float4_e1m2"):
        return _cast_fp4(x, dtype_name)
    return _cast_fp8(x, dtype_name)


def _cast_fp4(x, dtype_name):
    positive_values = {
        "float4_e2m1": np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
        ),
        "float4_e1m2": np.array(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32
        ),
    }[dtype_name]
    x = np.asarray(x, dtype=np.float32)
    absolute = np.nan_to_num(
        np.abs(x), nan=positive_values[-1], posinf=positive_values[-1]
    )
    distances = np.abs(absolute[..., None] - positive_values)
    min_distance = np.min(distances, axis=-1, keepdims=True)
    candidates = distances == min_distance
    even_codes = (np.arange(positive_values.size) & 1) == 0
    even_candidates = candidates & even_codes
    indices = np.argmax(
        np.where(
            np.any(even_candidates, axis=-1, keepdims=True), even_candidates, candidates
        ),
        axis=-1,
    )
    quantized = positive_values[indices]
    quantized = np.copysign(quantized, x).astype(np.float32)
    dtype = _get_ml_dtype(dtype_name)
    return quantized.astype(dtype) if dtype is not None else quantized


def _cast_fp8(x, dtype_name):
    dtype_name = _normalize_dtype_name(dtype_name)
    if dtype_name == "float8_e5m2":
        return _cast_fp8_from_bytes(
            _float_to_fp8_bytes(
                x, exp_bits=5, mant_bits=2, bias=15, max_exp_field=30, max_mant=3
            ),
            dtype_name,
        )
    return _cast_fp8_from_bytes(
        _float_to_fp8_bytes(
            x, exp_bits=4, mant_bits=3, bias=7, max_exp_field=15, max_mant=6
        ),
        dtype_name,
    )


def _float_to_fp8_bytes(x, *, exp_bits, mant_bits, bias, max_exp_field, max_mant):
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x)
    ax = np.abs(x)
    sign_bit = sign.astype(np.uint8) << 7
    out = sign_bit.copy()
    finite = np.isfinite(ax)
    min_subnormal = np.float32(2.0 ** (1 - bias - mant_bits))
    min_normal = np.float32(2.0 ** (1 - bias))

    sub_mask = finite & (ax > 0) & (ax < min_normal)
    sub_mant = np.rint(ax[sub_mask] / min_subnormal).astype(np.int32)
    sub_mant = np.clip(sub_mant, 0, (1 << mant_bits) - 1)
    out[sub_mask] = sign_bit[sub_mask] | sub_mant.astype(np.uint8)

    normal_mask = finite & (ax >= min_normal)
    normal_vals = ax[normal_mask]
    exp_unbiased = np.floor(np.log2(normal_vals)).astype(np.int32)
    scaled = normal_vals / np.exp2(exp_unbiased)
    mant = np.rint((scaled - 1.0) * (1 << mant_bits)).astype(np.int32)
    carry_mask = mant >= (1 << mant_bits)
    exp_unbiased[carry_mask] += 1
    mant[carry_mask] = 0
    exp_field = exp_unbiased + bias
    over_mask = (exp_field > max_exp_field) | (
        (exp_field == max_exp_field) & (mant > max_mant)
    )
    exp_field = np.where(over_mask, max_exp_field, exp_field)
    mant = np.where(over_mask, max_mant, mant)
    exp_field = np.clip(exp_field, 0, max_exp_field)
    mant = np.clip(mant, 0, (1 << mant_bits) - 1)
    out[normal_mask] = (
        sign_bit[normal_mask]
        | (exp_field.astype(np.uint8) << mant_bits)
        | mant.astype(np.uint8)
    )

    special_mask = ~finite
    out[special_mask] = (
        sign_bit[special_mask]
        | (np.uint8(max_exp_field) << mant_bits)
        | np.uint8(max_mant)
    )
    return out


def _cast_fp8_from_bytes(byte_array, dtype_name):
    dtype = _get_ml_dtype(dtype_name)
    if dtype is None:
        return byte_array.astype(np.float32)
    return byte_array.reshape(-1).view(dtype).reshape(byte_array.shape)


def _astype(x, dtype_name):
    dtype_name = _normalize_dtype_name(dtype_name)
    try:
        return x.astype(dtype_name)
    except (TypeError, ValueError):
        dtype = _get_ml_dtype(dtype_name)
        if dtype is not None:
            return x.astype(dtype)
    return x.astype(np.float32)


def _normalize_dtype_name(dtype_name):
    if not isinstance(dtype_name, str):
        return str(dtype_name)
    dtype_name = dtype_name.lower()
    if dtype_name == "float8_e4m3fnuz":
        return "float8_e4m3fn"
    return dtype_name


def _get_ml_dtype(dtype_name):
    if ml_dtypes is None:
        return None
    candidates = [dtype_name]
    if dtype_name == "float8_e8m0":
        candidates.extend(["float8_e8m0fnu", "float8_e8m0"])
    if dtype_name == "float4_e2m1":
        candidates.extend(["float4_e2m1fn", "float4_e2m1"])
    for candidate in candidates:
        dtype = getattr(ml_dtypes, candidate, None)
        if dtype is not None:
            return dtype
    return None
