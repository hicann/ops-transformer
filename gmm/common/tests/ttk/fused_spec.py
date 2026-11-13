# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared TTK support for GMM FinalizeRouting and SwiGLU quantization."""

import math
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


_CODEC = run_path(str(Path(__file__).with_name("tensor_codec.py")))


_nd, _cast, _numpy, _decode, _weights = (
    _CODEC[name] for name in ("_nd", "_cast", "_numpy", "_decode", "_weights")
)


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
    max_exp = max(fp8_max_exp, max_exp)
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


def _get_dtype_max(dtype_name):
    dtype_max = {
        "float8_e5m2": FP8_E5M2_MAX,
        "float8_e4m3fn": FP8_E4M3_MAX,
        "float4_e2m1": FP4_E2M1_MAX,
        "float4_e1m2": FP4_E1M2_MAX,
    }
    if dtype_name not in dtype_max:
        raise ValueError(f"unsupported y dtype: {dtype_name}")
    return dtype_max[dtype_name]


def _get_dtype_max_exp(dtype_name):
    dtype_max_exp = {
        "float8_e5m2": FP8_E5M2_MAX_EXP,
        "float8_e4m3fn": FP8_E4M3_MAX_EXP,
        "float4_e2m1": FP4_E2M1_MAX_EXP,
        "float4_e1m2": FP4_E1M2_MAX_EXP,
    }
    if dtype_name not in dtype_max_exp:
        raise ValueError(f"unsupported y dtype: {dtype_name}")
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
    maximum = _get_dtype_max(dtype_name)
    return np.clip(np.asarray(x, np.float32), -maximum, maximum).astype(
        _get_ml_dtype(dtype_name)
    )


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
    import en_dtypes

    native = getattr(en_dtypes, dtype_name, None)
    if native is not None:
        return native
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


def array(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [array(v, dtype) for v in value]
    if hasattr(value, "detach"):
        if "float8_e8m0" in str(value.dtype):
            return value.detach().float().cpu().numpy()
        value = _numpy(value)
    a = np.asarray(value)
    if dtype in ("float8_e8m0", "hifloat8") and a.dtype in (np.int8, np.uint8):
        import en_dtypes

        a = np.ascontiguousarray(a).view(getattr(en_dtypes, dtype))
    elif dtype and dtype.startswith("float4") and a.dtype in (np.uint8, np.int8):
        import en_dtypes

        codes = np.stack((a.astype(np.uint8) & 15, a.astype(np.uint8) >> 4), -1)
        a = codes.reshape(a.shape[:-1] + (a.shape[-1] * 2,)).view(
            getattr(en_dtypes, dtype)
        )
    elif dtype == "int4" and a.dtype == np.int32:
        bits = np.ascontiguousarray(a).view(np.uint32)
        codes = (
            bits[..., None].astype(np.uint64) >> (np.arange(8, dtype=np.uint64) * 4)
        ) & 15
        a = np.where(codes >= 8, codes.astype(np.int32) - 16, codes).reshape(
            a.shape[:-1] + (a.shape[-1] * 8,)
        )
    return a


def matrix(weight, fmt, trans, shape, dtype):
    if isinstance(weight, (list, tuple)):
        if len(weight) != 1:
            matrices = []
            source_shape = (shape[2], shape[1]) if trans else shape[1:]
            for tensor in weight:
                decoded = _nd(
                    array(tensor, dtype),
                    "FRACTAL_NZ" if fmt == "FRACTAL_NZ_C0_32" else fmt,
                    source_shape,
                )
                matrices.append(decoded.swapaxes(-1, -2) if trans else decoded)
            return np.stack(matrices)
        weight = weight[0]
    a = array(weight, dtype)
    if fmt in ("FRACTAL_NZ", "FRACTAL_NZ_C0_32"):
        if a.ndim < 5:
            # ACLNN exposes a logical view backed by a padded NZ allocation.
            raw = _numpy(
                weight.as_strided(
                    (weight.untyped_storage().nbytes() // weight.element_size(),),
                    (1,),
                    0,
                )
            )
            e, k, n = shape
            rows, cols = (n, k) if trans else (k, n)
            c0 = 64 if dtype.startswith("float4") or dtype == "int4" else 32
            a = array(raw, dtype).reshape(
                e, math.ceil(cols / c0), math.ceil(rows / 16), 16, c0
            )
        source = (shape[0], shape[2], shape[1]) if trans else shape
        a = _nd(a, "FRACTAL_NZ" if fmt == "FRACTAL_NZ_C0_32" else fmt, source)
    return a.swapaxes(-1, -2) if trans else a


def mm(x, w, xs, ws, groups, group_type=1):
    x = np.asarray(x).astype(np.float32)
    w = np.asarray(w).astype(np.float32)
    counts = (
        np.asarray(groups, np.int64) if group_type == 1 else np.diff(np.r_[0, groups])
    )
    result = np.zeros((x.shape[0], w.shape[-1]), np.float32)
    start = 0
    for e, count in enumerate(counts):
        stop = start + int(count)
        if count == 0:
            continue
        if np.asarray(xs).ndim == 3:
            left = (
                x[start:stop]
                * np.repeat(xs[start:stop].reshape(int(count), -1), 32, axis=1)[
                    :, : x.shape[1]
                ]
            )
            right_scale = ws[e].transpose(0, 2, 1).reshape(-1, w.shape[-1])
            right = w[e] * np.repeat(right_scale, 32, axis=0)[: w.shape[1]]
            result[start:stop] = left @ right
        else:
            result[start:stop] = (
                (x[start:stop] @ w[e]) * np.asarray(xs)[start:stop, None] * ws[e]
            )
        start = stop
    return result


def quantize(value, dtype, scale_alg=0, dst_type_max=0):
    value = _to_bfloat16_float32(value)
    if scale_alg == 0:
        y, scales = _dynamic_quant_mx_ocp(value, dtype)
    elif scale_alg == 1:
        y, scales = _dynamic_quant_mx(value, dtype)
    elif scale_alg == 2 and dtype == "float4_e2m1":
        y, scales = _dynamic_quant_mx_dynamic_dtype_range(value, dst_type_max)
    else:
        raise ValueError("Unsupported MX quantization output/scale algorithm")
    return [_cast_mx(y, dtype), _astype(scales, "float8_e8m0")]


def swiglu(value, mode=0, clamp=7, alpha=1.702, bias=1):
    left, right = np.split(value.astype(np.float32), 2, axis=1)
    if mode == 2:
        left = np.minimum(left, clamp)
        right = np.clip(right, -clamp, clamp) + bias
    elif mode != 0:
        raise ValueError("Only standard SwiGLU and the mode-2 interface are covered")
    else:
        alpha = 1
    z = np.exp(-np.abs(alpha * left))
    sigmoid = np.where(alpha * left >= 0, 1 / (1 + z), z / (1 + z))
    return left * sigmoid * right


def routing(
    result, logits, indices, batch_size, dtype, residual=None, residual_weight=1
):
    """Preserve the BF16 multiply and atomic-add rounding boundaries."""
    output = np.zeros((batch_size, result.shape[1]), np.float32)
    if residual is not None:
        output += np.asarray(residual, np.float32) * np.float32(residual_weight)
    if dtype == "bfloat16":
        bf = _to_bfloat16_float32
        output = bf(output)
        weighted = bf(bf(result) * bf(logits)[:, None])
        for index, value in zip(indices, weighted):
            output[index] = bf(output[index] + value)
    else:
        np.add.at(output, indices, result * logits[:, None])
    return _cast(output, dtype)


def routing_order_matches(actual, trace, residual_weight, rtol, atol):
    """Each value must match a legal BF16 atomic accumulation order."""
    import itertools

    bf = _to_bfloat16_float32
    weighted = bf(bf(trace["result"]) * bf(trace["logits"])[:, None])
    residual = (
        None
        if trace["residual"] is None
        else bf(trace["residual"] * np.float32(residual_weight))
    )
    valid = np.zeros(actual.shape, dtype=bool)
    for row in range(actual.shape[0]):
        terms = list(weighted[trace["indices"] == row])
        if residual is not None:
            terms.append(residual[row])
        if len(terms) > 6:
            raise ValueError(
                "Exhaustive BF16 atomic-order reference supports up to six terms per row"
            )
        for permutation in itertools.permutations(terms):
            value = np.zeros(actual.shape[1], np.float32)
            for term in permutation:
                value = bf(value + term)
            valid[row] |= np.isclose(
                actual[row], value, rtol=rtol, atol=atol, equal_nan=True
            )
    return valid


def _copy(target, value):
    if hasattr(target, "copy_"):
        import torch

        target.copy_(torch.as_tensor(value).to(target.dtype))
    else:
        target[...] = value


def tensor_args(args, meta):
    """Hooks receive the real API call, including keyword-only tensors."""
    if meta["f_route"] != "e2e":
        return args
    values = dict(zip(meta["f_pos_names"], args))
    values.update(
        {name: meta.get(name) for name in meta["f_param_names"] if name not in values}
    )
    return tuple(values[name] for name in meta["f_param_names"])


def _hook_customize(*args, **meta):
    """Generate legal expert partitions, MX encodings and routing permutations."""
    args = tensor_args(args, meta)
    route, family = meta["f_route"], meta["f_family"]
    ports = meta["f_ports"]
    for key in ("xs", "ws"):
        if key not in ports:
            continue
        seq = args[ports[key]]
        if seq is None:
            continue
        seq = seq if isinstance(seq, (list, tuple)) else [seq]
        for scale_index, value in enumerate(seq):
            if meta["f_sd"] == "float8_e8m0":
                exponents = (
                    (
                        np.arange(
                            value.numel() if hasattr(value, "numel") else value.size
                        )
                        + scale_index
                    )
                    % 3
                    - 1
                ).reshape(tuple(value.shape))
                data = np.exp2(exponents).astype(np.float32)
                if str(value.dtype).endswith(("uint8", "int8")):
                    data = (exponents + 127).astype(np.uint8)
                _copy(value, data)
    if route == "e2e" and meta["f_types"][0] == "hifloat8":
        import en_dtypes

        for key in ("x", "w"):
            values = args[ports[key]]
            for value in values if isinstance(values, (list, tuple)) else [values]:
                count = value.numel() if hasattr(value, "numel") else value.size
                codes = (np.arange(count) % 9 - 4).reshape(tuple(value.shape)).astype(
                    np.float32
                ) * 0.25
                _copy(value, codes.astype(en_dtypes.hifloat8).view(np.uint8))
    counts = np.array(meta["f_counts"], np.int64)
    if meta["f_gltype"] == 0:
        counts = np.cumsum(counts)
    if meta.get("ttk_group_list") is not None:
        counts = np.asarray(meta["ttk_group_list"], dtype=np.int64)
    _copy(args[ports["groups"]], counts)
    if family == "routing":
        _copy(
            args[ports["index"]],
            (
                np.arange(sum(meta["f_counts"]), dtype=np.int64)[::-1] % meta["f_bs"]
            ).copy(),
        )
    return None if route == "aclnn" else args


def _with_transposed_storage(tensor, dim0, dim1):
    """Give a dense CSV tensor transposed strides without changing logical values."""
    import torch

    source = tensor.transpose(dim0, dim1).clone(memory_format=torch.contiguous_format)
    return source.transpose(dim0, dim1)


def _hook_preprocess(*args, **meta):
    trans = meta["f_trans"] and not meta.get("f_graph_source")
    nd = meta["f_format"] == "ND"
    swiglu_trans = trans and meta["f_family"] == "swiglu"
    if nd and not swiglu_trans:
        return
    import torch
    import torch_npu

    args = tensor_args(args, meta)
    if swiglu_trans and meta["f_sd"] == "float8_e8m0":
        scales = args[meta["f_ports"]["ws"]]
        for i, scale in enumerate(scales):
            scales[i] = _with_transposed_storage(scale, -3, -2)
    port = meta["f_ports"]["w"]
    value = args[port]
    listed = isinstance(value, (list, tuple))
    tensors = value if listed else [value]
    results = []
    wd = meta["f_types"][1]
    for tensor in tensors:
        if nd:
            results.append(_with_transposed_storage(tensor, -1, -2))
            continue
        source = tensor.transpose(-1, -2).contiguous() if trans else tensor
        if wd.startswith("float4") and meta["f_format"] == "FRACTAL_NZ_C0_32":
            hint = 296 if wd == "float4_e2m1" else 297
            converted = torch_npu.npu_format_cast(
                source, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=hint
            )
        elif wd == "int4":
            decoded = torch.from_numpy(
                np.ascontiguousarray(_decode(tensor.cpu(), "int4"))
            ).to(tensor.device)
            source = decoded.transpose(-1, -2).contiguous() if trans else decoded
            converted = torch_npu.npu_convert_weight_to_int4pack(
                torch_npu.npu_format_cast(source, 29, customize_dtype=torch.int8)
            )
        else:
            converted = torch_npu.npu_format_cast(source, 29)
        results.append(converted.transpose(-1, -2) if trans else converted)
    if listed:
        for i, tensor in enumerate(results):
            value[i] = tensor
        return
    value.data = results[0]
    return


def attr(meta, snake, camel, default):
    value = meta.get(snake, meta.get(camel))
    return default if value is None else value


def _hook_golden(*args, _routing_trace=None, **meta):
    args = tensor_args(args, meta)
    ports, route = meta["f_ports"], meta["f_route"]
    family = meta["f_family"]
    xd, wd = meta["f_types"]
    x = array(args[ports["x"]], xd)
    if isinstance(x, list):
        x = x[0]
    groups = array(args[ports["groups"]])
    ws = array(args[ports["ws"]], meta["f_sd"])
    xs = array(args[ports["xs"]], meta["f_sd"]) if "xs" in ports else None
    if isinstance(ws, list):
        ws = np.stack(ws) if len(ws) > 1 else ws[0]
    value = args[ports["w"]]
    if route == "e2e":
        if family == "swiglu" and meta["f_trans"] and not meta.get("f_graph_source"):
            value = [_with_transposed_storage(v, -1, -2) for v in value]
        if isinstance(value, (list, tuple)):
            w = (
                np.stack([_decode(v, wd) for v in value])
                if len(value) > 1
                else _decode(value[0], wd)
            )
        else:
            w = _decode(value, wd)
    elif route == "aclnn":
        values = value if isinstance(value, (list, tuple)) else [value]
        if family == "routing" and meta["f_trans"]:
            values = [v.swapaxes(-1, -2) for v in values]
        c0 = meta["f_c0"] if meta["f_format"] != "ND" else 0
        decoded = _weights(
            values,
            wd,
            meta["f_trans"],
            c0,
            "int4" if wd == "int4" else "",
            meta["f_shape"][1],
            meta["f_shape"][2],
        )
        w = np.stack(decoded) if len(decoded) > 1 else decoded[0]
    else:
        w = matrix(value, meta["f_format"], meta["f_trans"], tuple(meta["f_shape"]), wd)
    if meta.get("f_graph_source"):
        w = w.swapaxes(-1, -2)
        ws = ws.swapaxes(-3, -2)
    if (
        meta["f_trans"]
        and np.asarray(ws).ndim == 4
        and (route == "kernel" or route == "aclnn" and family == "routing")
    ):
        ws = ws.transpose(0, 2, 1, 3)
    if xs is None:
        xs = np.ones(x.shape[0], np.float32)
    result = mm(x, w, xs, ws, groups, meta["f_gltype"])
    if family == "swiglu":
        activated = swiglu(
            result,
            attr(meta, "swiglu_mode", "swigluMode", meta.get("f_mode", 0)),
            clamp=attr(meta, "clamp_limit", "clampLimit", 7.0),
            alpha=attr(meta, "glu_alpha", "gluAlpha", 1.702),
            bias=attr(meta, "glu_bias", "gluBias", 1.0),
        )
        if meta["f_sd"] != "float8_e8m0":
            if meta.get("dequant_dtype") == 5:
                activated = activated.astype(np.float16).astype(np.float32)
            elif meta.get("dequant_dtype") == 15:
                activated = _to_bfloat16_float32(activated)
            limits = {
                "int8": 127,
                "float8_e4m3fn": 448,
                "float8_e5m2": 57344,
                "hifloat8": 32768,
            }
            limit = limits[meta["f_out"]]
            scale = np.max(np.abs(activated), axis=1) / np.float32(limit)
            safe = np.where(scale == 0, 1, scale)
            y = activated / safe[:, None]
            y = (
                np.clip(np.rint(y), -128, 127).astype(np.int8)
                if meta["f_out"] == "int8"
                else _cast(y, meta["f_out"])
            )
            return [y, scale.astype(np.float32)]
        return quantize(
            activated,
            meta["f_out"],
            attr(meta, "scale_alg", "scaleAlg", meta.get("f_alg", 0)),
            attr(meta, "dst_type_max", "dstTypeMax", meta.get("f_dstmax", 0)),
        )
    if "bias" in ports and args[ports["bias"]] is not None:
        b = array(args[ports["bias"]])
        counts = np.asarray(meta["f_counts"])
        result += np.repeat(b, counts, axis=0)
    logits = array(args[ports["logit"]]).astype(np.float32)
    indices = array(args[ports["index"]]).astype(np.int64)
    residual = (
        array(args[ports["residual"]]).astype(np.float32)
        if "residual" in ports and args[ports["residual"]] is not None
        else None
    )
    if _routing_trace is not None:
        _routing_trace.update(
            result=result, logits=logits, indices=indices, residual=residual
        )
    return [
        routing(
            result,
            logits,
            indices,
            meta["f_bs"],
            meta["f_out"],
            residual,
            meta.get("f_resweight", 1),
        )
    ]


def _hook_pre_compare(*outputs):
    """Decode byte carriers using the independent golden's semantic dtype."""
    import en_dtypes

    count = len(outputs) // 2
    values = list(outputs)
    for i, golden_value in enumerate(outputs[count:]):
        actual = outputs[i]
        if not isinstance(actual, np.ndarray) or actual.dtype not in (
            np.dtype("int8"),
            np.dtype("uint8"),
        ):
            continue
        name = str(golden_value.dtype)
        if name.startswith("float4_"):
            codes = np.ascontiguousarray(actual).view(np.uint8)
            if actual.shape != golden_value.shape:
                codes = np.stack((codes & 15, codes >> 4), axis=-1).reshape(
                    golden_value.shape
                )
            values[i] = codes.view(getattr(en_dtypes, name))
        elif name in ("hifloat8", "float8_e8m0"):
            values[i] = (
                np.ascontiguousarray(actual)
                .view(np.uint8)
                .view(getattr(en_dtypes, name))
            )
    return values


def _hook_compare_routing(output, expected, *, compare_context):
    """Validate BF16 atomic orders explicitly; keep the configured tolerances."""
    import ast

    meta = dict(compare_context.attributes)
    fields = compare_context.csv_fields
    rtol, ptol = ast.literal_eval(
        fields.get("precision_tolerances") or "((0.001,0.001),)"
    )[0]
    atol = ast.literal_eval(fields.get("absolute_precision") or "(0.001,)")[0]
    if output.shape != expected.shape:
        return {
            "pass": False,
            "precision": 0,
            "error_info": "routing output shape mismatch",
        }
    actual = array(output).astype(np.float32)
    if meta["f_out"] != "bfloat16":
        valid = np.isclose(
            actual,
            array(expected).astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    else:
        tensors = compare_context.input_tensors
        if meta["f_route"] == "e2e":
            meta.update(zip(meta["f_param_names"], tensors))
            args = tuple(meta[name] for name in meta["f_pos_names"])
        else:
            args = tuple(tensors)
        trace = {}
        _hook_golden(*args, _routing_trace=trace, **meta)
        valid = routing_order_matches(
            actual, trace, meta.get("f_resweight", 1), rtol, atol
        )
    bad = int(np.count_nonzero(~valid))
    return {
        "pass": bad <= valid.size * ptol,
        "precision": (1 - bad / max(1, valid.size)) * 100,
        "metrics": {
            "bad_elements": bad,
            "total_elements": int(valid.size),
            "bf16_atomic_orders": meta["f_out"] == "bfloat16",
        },
    }


_NAMES = {
    ("routing", "aclnn"): (
        "x1",
        "x2",
        "scale",
        "bias",
        "offsetOptional",
        "antiquantScaleOptional",
        "antiquantOffsetOptional",
        "pertokenScaleOptional",
        "groupList",
        "sharedInput",
        "logit",
        "rowIndex",
        "out",
    ),
    ("routing", "e2e"): (
        "x",
        "w",
        "group_list",
        "scale",
        "bias",
        "offset",
        "pertoken_scale",
        "shared_input",
        "logit",
        "row_index",
    ),
    ("swiglu", "aclnn"): (
        "x",
        "weight",
        "weightScale",
        "weightAssistMatrix",
        "bias",
        "xScale",
        "smoothScale",
        "groupList",
        "output",
        "outputScale",
    ),
    ("swiglu", "e2e"): (
        "x",
        "weight",
        "weight_scale",
        "x_scale",
        "group_list",
        "smooth_scale",
        "weight_assist_matrix",
        "bias",
    ),
}
_PORTS = {
    ("routing", "kernel"): dict(
        x=0, w=1, ws=2, bias=3, xs=4, groups=5, residual=6, logit=7, index=8
    ),
    ("routing", "aclnn"): dict(
        x=0, w=1, ws=2, bias=3, xs=7, groups=8, residual=9, logit=10, index=11
    ),
    ("routing", "e2e"): dict(
        x=0, w=1, groups=2, ws=3, bias=4, xs=6, residual=7, logit=8, index=9
    ),
    ("swiglu", "kernel"): dict(x=0, xs=1, groups=2, w=3, ws=4, bias=6),
    ("swiglu", "aclnn"): dict(x=0, w=1, ws=2, bias=4, xs=5, groups=7),
    ("swiglu", "e2e"): dict(x=0, w=1, ws=2, xs=3, groups=4, bias=7),
}


def _reference_metadata(family, route, data):
    if data is None:
        raise ValueError("missing ttk_reference in attributes1")
    names = _NAMES[family, "e2e"]
    return {
        **{f"f_{key}": value for key, value in data.items()},
        "f_family": family,
        "f_route": route,
        "f_ports": _PORTS[family, route],
        "f_param_names": names,
        "f_pos_names": names[: 3 if family == "routing" else 5],
    }


def make_spec(family, route):
    def customize(*args, ttk_reference=None, **kwargs):
        meta = _reference_metadata(family, route, ttk_reference)
        return _hook_customize(*args, **kwargs, **meta)

    def golden(*args, ttk_reference=None, **kwargs):
        meta = _reference_metadata(family, route, ttk_reference)
        if route == "aclnn" and family == "swiglu" and meta["f_mode"] == 2:
            names = (
                "swigluMode",
                "clampLimit",
                "gluAlpha",
                "gluBias",
                "roundMode",
                "scaleAlg",
                "dstTypeMax",
            )
            kwargs.update(zip(names, args[13:20], strict=True))
        return _hook_golden(*args, **kwargs, **meta)

    def preprocess(*args, ttk_reference=None, **kwargs):
        meta = _reference_metadata(family, route, ttk_reference)
        return _hook_preprocess(*args, **kwargs, **meta)

    def compare(output, expected, *, compare_context):
        from dataclasses import replace

        meta = _reference_metadata(
            family, route, compare_context.attributes.get("ttk_reference")
        )
        comparison = replace(
            compare_context, attributes={**compare_context.attributes, **meta}
        )
        return _hook_compare_routing(output, expected, compare_context=comparison)

    members = dict(
        golden=staticmethod(golden),
        customize_inputs=staticmethod(customize),
        pre_compare=staticmethod(_hook_pre_compare),
    )
    if route == "e2e":
        members["npu_preprocess"] = staticmethod(preprocess)
    if family == "routing":
        members["compare"] = staticmethod(compare)
    return type("FusedSpec", (), members)
