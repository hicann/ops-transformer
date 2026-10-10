# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent NumPy reference for GroupedMatmulQuant W4A16."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PrecisionResult:
    passed: bool
    matched_ratio: float
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    max_abs_error_limit: float


def pack_int4_to_int32(weight: np.ndarray) -> np.ndarray:
    """Pack logical signed INT4 weight [G, K, N] into [G,K/16,N/16,16,2]."""
    weight = np.asarray(weight)
    if weight.ndim != 3:
        raise ValueError(f"weight must have rank 3, got {weight.ndim}")
    group_num, k, n = weight.shape
    if k % 16 != 0 or n % 16 != 0:
        raise ValueError(f"K and N must be divisible by 16, got K={k}, N={n}")
    if np.any(weight < -8) or np.any(weight > 7):
        raise ValueError("signed INT4 values must be in [-8, 7]")

    # Logical [G,K,N] -> fractal [G,K1,N1,K0,N0]. Each INT32 stores
    # eight adjacent N0 nibbles, least-significant nibble first.
    tiled = weight.reshape(group_num, k // 16, 16, n // 16, 16).transpose(0, 1, 3, 2, 4)
    nibbles = np.bitwise_and(tiled.astype(np.int32), 0xF).astype(np.uint32)
    nibbles = nibbles.reshape(group_num, k // 16, n // 16, 16, 2, 8)
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 1, 1, 1, 8)
    packed = np.bitwise_or.reduce(nibbles << shifts, axis=-1).astype(np.uint32)
    return packed.view(np.int32)


def unpack_int32_to_int4(packed: np.ndarray) -> np.ndarray:
    """Unpack [G,K1,N1,16,2] INT32 storage into logical [G,K,N] INT8."""
    packed = np.asarray(packed)
    if packed.ndim != 5 or packed.shape[-2:] != (16, 2):
        raise ValueError(
            f"packed weight must have shape [G,K1,N1,16,2], got {packed.shape}"
        )
    group_num, k1, n1, _, _ = packed.shape
    words = packed.astype(np.int32, copy=False).view(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 1, 1, 1, 8)
    nibbles = ((words[..., None] >> shifts) & 0xF).astype(np.int8)
    signed = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.int8)
    tiled = signed.reshape(group_num, k1, n1, 16, 16)
    return tiled.transpose(0, 1, 3, 2, 4).reshape(group_num, k1 * 16, n1 * 16)


def round_to_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 (round-to-nearest-even), returned as float32."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)


def dequantize_weight(
    packed: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    scale_group_size: int,
    output_dtype: str,
) -> np.ndarray:
    """Apply the kernel's per-K-group `(q + offset) * scale` dequantization."""
    quantized = unpack_int32_to_int4(packed)
    group_num, k, n = quantized.shape
    if scale_group_size <= 0 or k % scale_group_size != 0:
        raise ValueError(
            f"scale_group_size={scale_group_size} must be positive and divide K={k}"
        )
    expected_shape = (group_num, k // scale_group_size, n)
    if tuple(scale.shape) != expected_shape or tuple(offset.shape) != expected_shape:
        raise ValueError(
            f"scale and offset must both have shape {expected_shape}, got {scale.shape} and {offset.shape}"
        )

    expanded_scale = np.repeat(np.asarray(scale), scale_group_size, axis=1)
    expanded_offset = np.repeat(np.asarray(offset), scale_group_size, axis=1)
    # The kernel first casts q to FP16 and performs q + offset in FP16 for
    # both output dtypes. The product is then rounded to the matmul dtype.
    shifted_fp16 = (
        quantized.astype(np.float16) + expanded_offset.astype(np.float16)
    ).astype(np.float16)
    if output_dtype == "float16":
        return (
            (shifted_fp16 * expanded_scale.astype(np.float16))
            .astype(np.float16)
            .astype(np.float32)
        )
    if output_dtype == "bfloat16":
        return round_to_bfloat16(
            shifted_fp16.astype(np.float32) * expanded_scale.astype(np.float32)
        )
    raise ValueError(f"unsupported output dtype: {output_dtype}")


def grouped_matmul_quant_golden(
    x: np.ndarray,
    packed: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    group_list: Optional[Sequence[int]],
    scale_group_size: int,
    output_dtype: str,
) -> np.ndarray:
    """Compute a float64 grouped-matmul reference from rounded operator inputs."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must have rank 2, got {x.ndim}")
    weight = dequantize_weight(packed, scale, offset, scale_group_size, output_dtype)
    group_num, k, n = weight.shape
    if x.shape[1] != k:
        raise ValueError(f"x K={x.shape[1]} does not match weight K={k}")

    if group_list is None:
        if group_num != 1:
            raise ValueError("group_list can be omitted only when G is 1")
        boundaries = [x.shape[0]]
    else:
        boundaries = [int(value) for value in group_list]
        if len(boundaries) != group_num:
            raise ValueError(
                f"group_list length must be G={group_num}, got {len(boundaries)}"
            )
        if any(value < 0 or value > x.shape[0] for value in boundaries):
            raise ValueError(
                f"group_list values must be in [0, M={x.shape[0]}], got {boundaries}"
            )
        if any(
            current < previous for previous, current in zip(boundaries, boundaries[1:])
        ):
            raise ValueError(f"group_list must be nondecreasing, got {boundaries}")
        if boundaries[-1] != x.shape[0]:
            raise ValueError(
                f"the last group_list value must equal M={x.shape[0]}, got {boundaries[-1]}"
            )

    output = np.zeros((x.shape[0], n), dtype=np.float64)
    start = 0
    for expert, end in enumerate(boundaries):
        if end > start:
            output[start:end] = x[start:end].astype(np.float64) @ weight[expert].astype(
                np.float64
            )
        start = end
    return output


def check_mixed_tolerance(
    actual: np.ndarray,
    golden: np.ndarray,
    output_dtype: str,
    required_matched_ratio: float = 0.99,
) -> PrecisionResult:
    """Apply the ecosystem mixed-tolerance rule for FP16/BF16 outputs."""
    thresholds = {
        "float16": (2.0**-9, 2.0**-9, 1.0e-1),
        "bfloat16": (2.0**-6, 2.0**-6, 1.0),
    }
    if output_dtype not in thresholds:
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    actual = np.asarray(actual, dtype=np.float64)
    golden = np.asarray(golden, dtype=np.float64)
    if actual.shape != golden.shape:
        raise ValueError(
            f"shape mismatch: actual={actual.shape}, golden={golden.shape}"
        )
    if actual.size == 0:
        rtol, atol, max_abs_error_limit = thresholds[output_dtype]
        return PrecisionResult(True, 1.0, 0.0, 0.0, atol, rtol, max_abs_error_limit)

    rtol, atol, max_abs_error_limit = thresholds[output_dtype]
    abs_error = np.abs(actual - golden)
    rel_error = abs_error / np.maximum(np.abs(golden), np.finfo(np.float64).tiny)
    finite = np.isfinite(actual) & np.isfinite(golden)
    matched = finite & (abs_error <= atol + rtol * np.abs(golden))
    matched_ratio = float(np.count_nonzero(matched) / matched.size)
    max_abs_error = float(np.max(abs_error))
    max_rel_error = float(np.max(rel_error))
    passed = (
        matched_ratio >= required_matched_ratio and max_abs_error <= max_abs_error_limit
    )
    return PrecisionResult(
        passed,
        matched_ratio,
        max_abs_error,
        max_rel_error,
        atol,
        rtol,
        max_abs_error_limit,
    )
