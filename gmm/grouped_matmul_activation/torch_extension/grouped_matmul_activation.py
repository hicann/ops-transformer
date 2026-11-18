# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import struct
from typing import List, Optional, Union

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


SWIGLU_ACT_TYPE = "swiglu"
SWIGLU_MODE_SPLIT = 2
DEFAULT_CLAMP_LIMIT = 7.0
DEFAULT_GLU_ALPHA = 1.702
DEFAULT_GLU_BIAS = 1.0
MATRIX_RANK = 2
WEIGHT_RANK = 3
WEIGHT_SCALE_RANK = 4
PACKED_FP4_FACTOR = 2
MX_GROUP_SIZE = 64
MX_SCALE_PAIR = 2
MX_ALIGNMENT = 128
WEIGHT_SCALE_N_DIM = 2
WEIGHT_COLUMN_DIM = 2
SCALE_PAIR_DIM = 3
SWIGLU_SPLIT_FACTOR = 2
GE_DTYPE_FLOAT4_E2M1 = 40
GE_DTYPE_FLOAT8_E8M0 = 37
ACL_DTYPE_OFFSET = 256


def _normalize_input_dtype(dtype, expected, name):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        native_name = (
            "float4_e2m1fn_x2" if expected == GE_DTYPE_FLOAT4_E2M1 else "float8_e8m0fnu"
        )
        if dtype != getattr(torch, native_name, None):
            raise ValueError(f"{name} must specify {native_name}.")
        return expected + ACL_DTYPE_OFFSET
    if not isinstance(dtype, int) or isinstance(dtype, bool):
        raise TypeError(f"{name} must be a dtype or an integer dtype enum.")
    logical_dtype = dtype - ACL_DTYPE_OFFSET if dtype >= ACL_DTYPE_OFFSET else dtype
    if logical_dtype != expected:
        raise ValueError(f"{name} specifies an unsupported logical dtype.")
    return logical_dtype + ACL_DTYPE_OFFSET


def _check_input_storage(tensor, dtype, expected, name):
    _normalize_input_dtype(dtype, expected, f"{name}_dtype")
    native_name = (
        "float4_e2m1fn_x2" if expected == GE_DTYPE_FLOAT4_E2M1 else "float8_e8m0fnu"
    )
    if tensor.dtype not in (torch.int8, torch.uint8, getattr(torch, native_name, None)):
        raise ValueError(f"{name} must use byte storage or {native_name} storage.")


def _require_tensor_list(value, name):
    if not isinstance(value, list) or not all(
        isinstance(item, torch.Tensor) for item in value
    ):
        raise TypeError(f"{name} must be a TensorList (list of Tensor).")
    return value


class GroupedMatmulActivationOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("grouped_matmul_activation", category="gmm")

    def sources(self):
        return ["csrc/gmm/grouped_matmul_activation.cpp"]

    def schema(self):
        return (
            "grouped_matmul_activation("
            "Tensor x, Tensor x_scale, Tensor group_list, Tensor[] weight, Tensor[] weight_scale, "
            "*, Tensor[]? bias=None, str activation_type='swiglu', "
            "int group_list_type=0, int[]? tuning_config=None, "
            "int swiglu_mode=2, float clamp_limit=7.0, float glu_alpha=1.702, float glu_bias=1.0, "
            "int? x_dtype=None, int? weight_dtype=None, int? x_scale_dtype=None, "
            "int? weight_scale_dtype=None) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_matmul_activation_meta(
            x,
            x_scale,
            group_list,
            weight,
            weight_scale,
            bias=None,
            activation_type=SWIGLU_ACT_TYPE,
            group_list_type=0,
            tuning_config=None,
            swiglu_mode=SWIGLU_MODE_SPLIT,
            clamp_limit=DEFAULT_CLAMP_LIMIT,
            glu_alpha=DEFAULT_GLU_ALPHA,
            glu_bias=DEFAULT_GLU_BIAS,
            x_dtype=None,
            weight_dtype=None,
            x_scale_dtype=None,
            weight_scale_dtype=None,
        ):
            if len(weight) != 1 or len(weight_scale) != 1:
                raise ValueError(
                    "weight and weight_scale must each contain exactly one tensor."
                )
            if bias is not None:
                _require_tensor_list(bias, "bias")
                if len(bias) != 0:
                    raise ValueError(
                        "bias must be None or an empty TensorList for the current A4W4 path."
                    )
            if activation_type != SWIGLU_ACT_TYPE:
                raise ValueError("only activation_type='swiglu' is supported.")
            if group_list_type not in (0, 1):
                raise ValueError("group_list_type must be 0 or 1.")
            if swiglu_mode != SWIGLU_MODE_SPLIT:
                raise ValueError("only swiglu_mode=2 is supported.")
            for name, value in (
                ("clamp_limit", clamp_limit),
                ("glu_alpha", glu_alpha),
                ("glu_bias", glu_bias),
            ):
                if (
                    not math.isfinite(value)
                    or abs(value) > torch.finfo(torch.float32).max
                ):
                    raise ValueError(
                        f"{name} must be finite and representable by float32."
                    )
            if struct.unpack("f", struct.pack("f", clamp_limit))[0] <= 0:
                raise ValueError(
                    "clamp_limit must remain positive after conversion to float32."
                )
            if tuning_config is not None and len(tuning_config) != 0:
                raise ValueError("tuning_config must be None or an empty list.")
            if (
                x.dim() != MATRIX_RANK
                or weight[0].dim() != WEIGHT_RANK
                or weight_scale[0].dim() != WEIGHT_SCALE_RANK
            ):
                raise ValueError(
                    "x must be 2D, weight[0] must be 3D and weight_scale[0] must be 4D."
                )
            for name, tensor, dtype, expected in (
                ("x", x, x_dtype, GE_DTYPE_FLOAT4_E2M1),
                ("x_scale", x_scale, x_scale_dtype, GE_DTYPE_FLOAT8_E8M0),
                ("weight[0]", weight[0], weight_dtype, GE_DTYPE_FLOAT4_E2M1),
                (
                    "weight_scale[0]",
                    weight_scale[0],
                    weight_scale_dtype,
                    GE_DTYPE_FLOAT8_E8M0,
                ),
            ):
                _check_input_storage(tensor, dtype, expected, name)
            if group_list.dtype != torch.int64 or group_list.dim() != 1:
                raise ValueError("group_list must be a 1D torch.int64 tensor.")
            logical_k = x.shape[1] * PACKED_FP4_FACTOR
            if logical_k <= 0 or logical_k % MX_ALIGNMENT != 0:
                raise ValueError(
                    "A4W4 requires logical K to be positive and divisible by 128."
                )
            logical_n = weight_scale[0].shape[WEIGHT_SCALE_N_DIM]
            if logical_n <= 0 or logical_n % MX_ALIGNMENT != 0:
                raise ValueError(
                    "A4W4 requires logical N to be positive and divisible by 128."
                )
            if x.shape[0] <= 0:
                raise ValueError("M must be positive.")
            if x_scale.shape != (x.shape[0], logical_k // MX_GROUP_SIZE, MX_SCALE_PAIR):
                raise ValueError("x_scale must have shape (M, K/64, 2).")
            experts = group_list.shape[0]
            if experts <= 0 or weight_scale[0].shape != (
                experts,
                logical_k // MX_GROUP_SIZE,
                logical_n,
                MX_SCALE_PAIR,
            ):
                raise ValueError("weight_scale[0] must have shape (E, K/64, N, 2).")
            transposed = (
                weight[0].stride(1) == 1
                and weight[0].stride(WEIGHT_COLUMN_DIM) == weight[0].shape[1]
            )
            expected_weight = (
                (experts, logical_k // PACKED_FP4_FACTOR, logical_n)
                if transposed
                else (experts, logical_k, logical_n // PACKED_FP4_FACTOR)
            )
            if weight[0].shape != expected_weight:
                raise ValueError(
                    "weight[0] shape must match the packed K/N dimensions and expert count."
                )
            scale_transposed = (
                weight_scale[0].stride(SCALE_PAIR_DIM) == 1
                and weight_scale[0].stride(1) == MX_SCALE_PAIR
                and weight_scale[0].stride(WEIGHT_SCALE_N_DIM)
                == weight_scale[0].shape[1] * MX_SCALE_PAIR
            )
            if transposed != scale_transposed:
                raise ValueError(
                    "weight and weight_scale must use consistent transpose views."
                )
            return torch.empty(
                (x.shape[0], logical_n // SWIGLU_SPLIT_FACTOR),
                dtype=torch.bfloat16,
                device="meta",
            )


_op_builder = GroupedMatmulActivationOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _grouped_matmul_activation(
    x,
    x_scale,
    group_list,
    weight,
    weight_scale,
    bias=None,
    activation_type=SWIGLU_ACT_TYPE,
    group_list_type=0,
    tuning_config=None,
    swiglu_mode=SWIGLU_MODE_SPLIT,
    clamp_limit=DEFAULT_CLAMP_LIMIT,
    glu_alpha=DEFAULT_GLU_ALPHA,
    glu_bias=DEFAULT_GLU_BIAS,
    x_dtype=None,
    weight_dtype=None,
    x_scale_dtype=None,
    weight_scale_dtype=None,
):
    return _op_builder.load().grouped_matmul_activation(
        x,
        x_scale,
        group_list,
        weight,
        weight_scale,
        bias,
        activation_type,
        group_list_type,
        tuning_config,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
        x_dtype,
        weight_dtype,
        x_scale_dtype,
        weight_scale_dtype,
    )


def grouped_matmul_activation(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    weight: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    *,
    bias: Optional[List[torch.Tensor]] = None,
    activation_type: str = SWIGLU_ACT_TYPE,
    group_list_type: int = 0,
    tuning_config: Optional[List[int]] = None,
    swiglu_mode: int = SWIGLU_MODE_SPLIT,
    clamp_limit: float = DEFAULT_CLAMP_LIMIT,
    glu_alpha: float = DEFAULT_GLU_ALPHA,
    glu_bias: float = DEFAULT_GLU_BIAS,
    x_dtype: Optional[Union[torch.dtype, int]] = None,
    weight_dtype: Optional[Union[torch.dtype, int]] = None,
    x_scale_dtype: Optional[Union[torch.dtype, int]] = None,
    weight_scale_dtype: Optional[Union[torch.dtype, int]] = None,
) -> torch.Tensor:
    """A4W4 WeightNZ grouped matmul followed by SwiGLU, with direct BF16 output."""
    bias_value = None if bias is None else _require_tensor_list(bias, "bias")
    x_dtype = _normalize_input_dtype(x_dtype, GE_DTYPE_FLOAT4_E2M1, "x_dtype")
    weight_dtype = _normalize_input_dtype(
        weight_dtype, GE_DTYPE_FLOAT4_E2M1, "weight_dtype"
    )
    x_scale_dtype = _normalize_input_dtype(
        x_scale_dtype, GE_DTYPE_FLOAT8_E8M0, "x_scale_dtype"
    )
    weight_scale_dtype = _normalize_input_dtype(
        weight_scale_dtype, GE_DTYPE_FLOAT8_E8M0, "weight_scale_dtype"
    )
    return torch.ops.cann_ops_transformer.grouped_matmul_activation(
        x,
        x_scale,
        group_list,
        _require_tensor_list(weight, "weight"),
        _require_tensor_list(weight_scale, "weight_scale"),
        bias=bias_value,
        activation_type=activation_type,
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        swiglu_mode=swiglu_mode,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
        x_dtype=x_dtype,
        weight_dtype=weight_dtype,
        x_scale_dtype=x_scale_dtype,
        weight_scale_dtype=weight_scale_dtype,
    )
