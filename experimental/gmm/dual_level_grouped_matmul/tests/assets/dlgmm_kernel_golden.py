#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Kernel/GEIR TestSpec with the dual-level Matmul numerical policy.

FP4 inputs are numerical ndarrays (including en_dtypes.float4_e2m1), not packed
byte files. E8M0 inputs may be en_dtypes arrays or raw uint8 scale codes.
The reference is shipped alongside this plugin; no sibling repository is needed.
"""

import importlib.util
from pathlib import Path

import numpy as np

_reference_path = Path(__file__).with_name("dlgmm_reference.py")
_module_spec = importlib.util.spec_from_file_location(
    "_dlgmm_dlqbmm_reference", _reference_path
)
_reference = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_reference)


def _single_tensor(value, name, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 1 or value[0] is None:
        raise ValueError(f"{name} must contain exactly one tensor")
    return value[0]


def _dtype_name(value):
    try:
        return np.dtype(value).name
    except (TypeError, ValueError):
        return str(value).removeprefix("torch.")


def _ge_output_dtype(value):
    name = _dtype_name(value)
    if name not in ("float16", "bfloat16"):
        raise ValueError("Output dtype must be float16 or bfloat16")
    return 1 if name == "float16" else 27


def _group_segments(grouplist, m, experts, group_list_type):
    values = np.asarray(grouplist)
    if group_list_type not in (0, 1, 2):
        raise ValueError("group_list_type must be 0, 1 or 2")
    expected = (experts, 2) if group_list_type == 2 else (experts,)
    if values.shape != expected or values.dtype != np.dtype("int64"):
        raise ValueError(f"grouplist must be INT64 with shape {expected}")
    segments, previous, empty_seen = [], 0, False
    for index, item in enumerate(values):
        expert, value = (
            (int(item[0]), int(item[1])) if group_list_type == 2 else (index, int(item))
        )
        if group_list_type == 2:
            if expert < 0 or expert >= experts or (empty_seen and value != 0):
                raise ValueError(
                    "Sparse expert index out of range or nonempty group follows an empty group"
                )
            empty_seen = empty_seen or value == 0
        value = int(value)  # Python accumulation avoids int64 overflow wrapping.
        end = previous + value if group_list_type != 0 else value
        if value < 0 or end < previous or end > m:
            raise ValueError("Invalid grouplist length/boundary")
        segments.append((expert, end))
        previous = end
    if previous != m:
        raise ValueError("grouplist must cover exactly M rows")
    return segments


def _group_ends(grouplist, m, experts, group_list_type):
    return [end for _, end in _group_segments(grouplist, m, experts, group_list_type)]


def _weight_nd(weight, expected_shape, kwargs):
    ori_shapes = kwargs.get("input_ori_shapes", ())
    ori_shape = ori_shapes[3] if len(ori_shapes) > 3 else None
    if ori_shape is not None and tuple(ori_shape) != expected_shape:
        raise ValueError(
            "weight input_ori_shapes disagrees with E/K/N and transpose_weight"
        )
    if weight.ndim == 5:
        weight = _reference.nz_to_nd(weight, expected_shape)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"weight must have logical shape {expected_shape}, got {weight.shape}"
        )
    return weight


def customize_inputs(
    x,
    x_level0_scale,
    x_level1_scale,
    weight,
    weight_level0_scale,
    weight_level1_scale,
    grouplist,
    bias=None,
    *,
    dtype=1,
    transpose_weight=False,
    group_list_type=0,
    tuning_config=(0,),
    **kwargs,
):
    # Do not replace an NZ storage tensor with ND data in the device input pool.
    # Group values come directly from attributes['grouplist'] in TTK.
    return (
        x,
        x_level0_scale,
        x_level1_scale,
        weight,
        weight_level0_scale,
        weight_level1_scale,
        grouplist,
        bias,
    )


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def golden(
    x,
    x_level0_scale,
    x_level1_scale,
    weight,
    weight_level0_scale,
    weight_level1_scale,
    grouplist,
    bias=None,
    *,
    dtype=1,
    transpose_weight=False,
    group_list_type=0,
    tuning_config=(0,),
    level0_group_size=512,
    level1_group_size=32,
    **kwargs,
):
    """FP32 partials per 32 K, FP32 scaling per 512 K, then bias/output cast."""
    if (level0_group_size, level1_group_size) != (512, 32):
        raise ValueError("Only group sizes (512,32) are supported")
    if dtype not in (1, 27):
        raise ValueError("dtype must be GE DT_FLOAT16=1 or DT_BF16=27")
    output_dtype = "float16" if dtype == 1 else "bfloat16"
    metadata = kwargs.get("output_dtypes")
    if metadata is not None and (
        len(metadata) != 1 or _dtype_name(metadata[0]) != output_dtype
    ):
        raise ValueError("output_dtypes must agree with the dtype attribute")
    if tuning_config is not None and tuple(tuning_config) != (0,):
        raise ValueError("Only default tuning_config=(0,) is supported")
    if x.ndim != 2 or weight_level0_scale.ndim != 3:
        raise ValueError("Expected x (M,K) and weight_level0_scale (E,P,N)")
    m, k = x.shape
    experts, _, n = weight_level0_scale.shape
    if (
        m <= 0
        or experts <= 0
        or k <= 2
        or k % 2
        or n <= 1
        or (not transpose_weight and n % 2)
    ):
        raise ValueError("Unsupported MXFP4 grouped Matmul dimensions")
    scale_pair_size = 2 * level1_group_size
    p = (k + level0_group_size - 1) // level0_group_size
    q = (k + scale_pair_size - 1) // scale_pair_size
    shapes = [
        (x_level0_scale, (m, p)),
        (x_level1_scale, (m, q, 2)),
        (weight_level0_scale, (experts, p, n)),
        (
            weight_level1_scale,
            (experts, n, q, 2) if transpose_weight else (experts, q, n, 2),
        ),
    ]
    for tensor, expected in shapes:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"scale shape {tensor.shape} must be {expected}")
    if bias is not None and tuple(bias.shape) != (experts, n):
        raise ValueError("bias must have shape (E,N) or be None")
    groups = _group_segments(grouplist, m, experts, group_list_type)
    expected_weight = (experts, n, k) if transpose_weight else (experts, k, n)
    weight = _weight_nd(weight, expected_weight, kwargs)
    parts, start = [], 0
    for expert, end in groups:
        if end != start:
            # The reference expects B1 in (N,Q,2) for its transposed-B contract.
            b1 = weight_level1_scale[expert]
            if not transpose_weight:
                b1 = b1.transpose(1, 0, 2)
            part = _reference.DualLevelMatmulReference.golden(
                x[start:end],
                weight[expert],
                x_level0_scale[start:end],
                x_level1_scale[start:end],
                weight_level0_scale[expert],
                b1,
                None if bias is None else bias[expert],
                dtype=dtype,
                transpose_x1=False,
                transpose_x2=transpose_weight,
                level0_group_size=level0_group_size,
                level1_group_size=level1_group_size,
            )[0]
            parts.append(part)
        start = end
    return [np.concatenate(parts, axis=0)]


class DualLevelGroupedMatmulTestSpec:
    tolerance = _reference.TOLERANCE
    golden = staticmethod(golden)
    customize_inputs = staticmethod(customize_inputs)
    pre_compare = staticmethod(pre_compare)
    compare = staticmethod(_reference.DualLevelMatmulReference.compare)


__spec__ = {"dual_level_grouped_matmul": "DualLevelGroupedMatmulTestSpec"}
