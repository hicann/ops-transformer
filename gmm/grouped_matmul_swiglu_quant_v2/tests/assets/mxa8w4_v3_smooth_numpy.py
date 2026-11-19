#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-only MXA8W4 V3 smoothing reference (numpy; no torch/NPU).

Inputs are raw uint8 E4M3FN/E8M0 bytes, and decoded logical ND weight
[E,K,N] in E2M1 values. WeightNZ physical tiling is intentionally outside
this numerical reference. Use unpack_fp4 for packed *logical ND* bytes;
low nibble comes first, and 0x22 decodes to [1,1], never [2,2].

scale_a: [M,ceil(K/64),2]; scale_b: [E,ceil(K/64),N,2].
smooth_scale: None or FP32 [E,N/2]. group_list_type: 0=cumsum, 1=counts.
Output and output_scale are raw uint8 E4M3FN and E8M0 bytes respectively.

Mode2 matches V3/kernel: a=min(left,clamp), then
out=a/(1+exp(-alpha*a))*(clip(right,-clamp,clamp)+bias).
Apply expert-wise FP32 smooth to out before BF16RNE rounding.
Finite inputs and normal-range quantization are the focused test domain.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FP4_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def bf16_rne(x):
    """BF16 round-to-nearest-even, returned as exactly widened FP32."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    special_bits = bits & np.uint32(0xFFFF0000)
    special_bits = np.where(
        np.isnan(x), special_bits | np.uint32(0x00400000), special_bits
    )
    return np.where(special, special_bits, rounded).astype(np.uint32).view(np.float32)


def _load_quant_helpers():
    """Reuse repository golden algorithms without importing its torch runtime.

    Select its numpy-only function definitions and constants via AST, retaining
    the actual algorithm bodies. Replace only the torch BF16 multiplication
    adapter with numpy RNE. No sys.modules stubs or changes to source files.
    """
    source = (
        Path(__file__).resolve().parents[3]
        / "grouped_matmul_activation_quant/tests/assets/golden.py"
    )
    reference = source.with_name("gmmaq_reference.py")
    if reference.is_file():
        source = reference
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    excluded = {
        "_bf16_mul_float32",
        "_to_bfloat16_float32",
        "_torch_matmul",
        "_gelu_tanh",
        "grouped_matmul_activation_quant_golden",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or (isinstance(node, ast.FunctionDef) and node.name not in excluded)
    ]
    namespace = {"np": np, "ml_dtypes": None}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    namespace["_bf16_mul_float32"] = lambda x, value: bf16_rne(
        bf16_rne(x) * bf16_rne(value)
    )
    return namespace


_QUANT = _load_quant_helpers()


def decode_e4m3(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint8:
        raise TypeError("E4M3 input must be raw uint8 bytes")
    exp = ((bits >> 3) & 15).astype(np.int32)
    mant = (bits & 7).astype(np.float32)
    value = np.where(
        exp == 0, mant * np.float32(2**-9), (1 + mant / 8) * np.exp2(exp - 7)
    ).astype(np.float32)
    value = np.where((exp == 15) & (mant == 7), np.float32(np.nan), value)
    return np.copysign(value, np.where(bits & 128, -1, 1)).astype(np.float32)


def decode_e8m0(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint8:
        raise TypeError("E8M0 input must be raw uint8 bytes")
    return np.where(bits == 255, np.nan, np.exp2(bits.astype(np.int32) - 127)).astype(
        np.float32
    )


def unpack_fp4(packed):
    """Unpack logical ND bytes along the last axis, low nibble first."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint8:
        raise TypeError("packed FP4 must be uint8")
    codes = np.stack((packed & 15, packed >> 4), axis=-1)
    values = np.copysign(FP4_VALUES[codes & 7], np.where(codes & 8, -1, 1))
    return values.astype(np.float32).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def pack_fp4(weight):
    """Pack exactly representable E2M1 ND values; reject lossy conversion."""
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape[-1] % 2 or not np.all(np.isin(np.abs(weight), FP4_VALUES)):
        raise ValueError(
            "weight needs even last dimension and exact finite E2M1 values"
        )
    codes = np.argmax(np.abs(weight)[..., None] == FP4_VALUES, axis=-1).astype(np.uint8)
    codes |= np.signbit(weight).astype(np.uint8) << 3
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def quantize_bf16(x, scale_alg):
    """Run existing golden scaleAlg=0/1 and E4M3 SAT/RNE conversion."""
    if scale_alg not in (0, 1):
        raise ValueError("scale_alg must be 0 or 1")
    algorithm = _QUANT[
        "_dynamic_quant_mx_ocp" if scale_alg == 0 else "_dynamic_quant_mx"
    ]
    y, scales = algorithm(np.asarray(x, dtype=np.float32), "float8_e4m3fn")
    y_bytes = _QUANT["_float_to_fp8_bytes"](
        y, exp_bits=4, mant_bits=3, bias=7, max_exp_field=15, max_mant=6
    )
    # Existing caster clamps subnormal mantissa 8 to 7, losing the carry into
    # the smallest normal. Correct that conversion boundary locally; keep
    # the repository's scale algorithms verbatim.
    carry = (np.abs(y) < np.float32(2**-6)) & (np.abs(y) >= np.float32(7.5 * 2**-9))
    y_bytes = np.where(carry, (y_bytes & np.uint8(128)) | np.uint8(8), y_bytes).astype(
        np.uint8
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_bytes = np.where(scales > 0, np.log2(scales) + 127, 0)
    scale_bytes = np.where(np.isnan(scales), 255, scale_bytes).astype(np.uint8)
    return y_bytes, scale_bytes


def mode2(mm, clamp_limit=7.0, alpha=1.702, bias=1.0):
    """Use the clamped activation in both the numerator and the exponent."""
    act, gate = np.split(np.asarray(mm, dtype=np.float32), 2, axis=-1)
    clamp, alpha, bias = map(np.float32, (clamp_limit, alpha, bias))
    if not np.all(np.isfinite([clamp, alpha, bias])) or clamp <= 0:
        raise ValueError("mode2 parameters must be finite; clamp must be positive")
    act = np.minimum(act, clamp)
    with np.errstate(over="ignore"):
        return (
            act
            / (np.float32(1) + np.exp(-alpha * act))
            * (np.clip(gate, -clamp, clamp) + bias)
        ).astype(np.float32)


@dataclass
class SmoothGoldenResult:
    output: np.ndarray
    output_scale: np.ndarray
    group_offsets: np.ndarray
    matmul: np.ndarray
    swiglu: np.ndarray
    smoothed: np.ndarray
    bf16: np.ndarray


def golden(
    x,
    weight,
    scale_a,
    scale_b,
    group_list,
    *,
    smooth_scale=None,
    group_list_type=0,
    scale_alg=0,
    clamp_limit=7.0,
    alpha=1.702,
    bias=1.0,
):
    """FP32 dequant/GMM -> mode2 -> FP32 smooth -> BF16RNE -> V3 MXFP8."""
    x = decode_e4m3(x)
    weight = np.asarray(weight, dtype=np.float32)
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError("expected x[M,K] and logical ND weight[E,K,N]")
    m, k = x.shape
    e, wk, n = weight.shape
    if k <= 0 or wk != k or n <= 0 or n % 2:
        raise ValueError("K must match and be positive; N must be positive and even")
    if not np.all(np.isin(np.abs(weight), FP4_VALUES)) or not np.all(np.isfinite(x)):
        raise ValueError("finite E4M3 x and exact E2M1 weight required")
    groups = np.asarray(group_list)
    if (
        groups.shape != (e,)
        or groups.dtype.kind not in "iu"
        or group_list_type not in (0, 1)
    ):
        raise ValueError("group_list must contain E integers; type must be 0 or 1")
    ends = (
        np.cumsum(groups, dtype=np.int64)
        if group_list_type == 1
        else groups.astype(np.int64)
    )
    offsets = np.concatenate(([0], ends))
    if np.any(np.diff(offsets) < 0) or offsets[-1] != m:
        raise ValueError(
            "groups must be monotonic/nonnegative and cover exactly M rows"
        )
    sa, sb = decode_e8m0(scale_a), decode_e8m0(scale_b)
    blocks = (k + 63) // 64
    if sa.shape != (m, blocks, 2) or sb.shape != (e, blocks, n, 2):
        raise ValueError("invalid scale_a/scale_b paired E8M0 shape")
    if not np.all(np.isfinite(sa)) or not np.all(np.isfinite(sb)):
        raise ValueError("finite input scales required")
    if smooth_scale is not None:
        smooth_scale = np.asarray(smooth_scale)
        if smooth_scale.dtype != np.float32 or smooth_scale.shape != (e, n // 2):
            raise ValueError("smooth_scale must be FP32 [E,N/2]")
        if not np.all(np.isfinite(smooth_scale)):
            raise ValueError("smooth_scale must be finite")
    xd = x * np.repeat(sa.reshape(m, blocks * 2), 32, axis=1)[:, :k]
    mm = np.zeros((m, n), dtype=np.float32)
    for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start == end:
            continue
        scales = sb[expert].transpose(0, 2, 1).reshape(blocks * 2, n)
        wd = weight[expert] * np.repeat(scales, 32, axis=0)[:k]
        mm[start:end] = xd[start:end] @ wd
    activated = mode2(mm, clamp_limit, alpha, bias)
    smoothed = activated.copy()
    if smooth_scale is not None:
        for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            smoothed[start:end] *= smooth_scale[expert]
    rounded = bf16_rne(smoothed)
    output, output_scale = quantize_bf16(rounded, scale_alg)
    return SmoothGoldenResult(
        output, output_scale, offsets, mm, activated, smoothed, rounded
    )
