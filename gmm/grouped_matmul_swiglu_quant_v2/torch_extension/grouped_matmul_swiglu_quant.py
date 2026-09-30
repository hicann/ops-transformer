# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


GE_DTYPE_FLOAT = 0
TORCH_DTYPE_FLOAT = 6
GE_DTYPE_INT8 = 1
GE_DTYPE_FLOAT8_E5M2 = 35
GE_DTYPE_FLOAT8_E4M3FN = 36
GE_DTYPE_FLOAT8_E8M0 = 37
ACL_DTYPE_OFFSET = 256
MX_GROUP_SIZE = 64
WEIGHT_SCALE_RANK = 4
WEIGHT_SCALE_N_AXIS = 2
SWIGLU_SPLIT_FACTOR = 2
MX_SCALE_PAIR_SIZE = 2
CEIL_DIV_ADJUSTMENT = 1
MX_QUANT_MODE = 2

FLOAT8_E8M0_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", torch.uint8)
)

_TORCH_DTYPE_TO_GE_DTYPE = {
    torch.float32: GE_DTYPE_FLOAT,
    torch.int8: GE_DTYPE_INT8,
    torch.float8_e5m2: GE_DTYPE_FLOAT8_E5M2,
    torch.float8_e4m3fn: GE_DTYPE_FLOAT8_E4M3FN,
    FLOAT8_E8M0_DTYPE: GE_DTYPE_FLOAT8_E8M0,
}


def _normalize_tensor_list(value, name):
    if not isinstance(value, list):
        raise TypeError(
            f"{name} must be a TensorList (list of Tensor), but got {type(value)}."
        )
    for index, tensor in enumerate(value):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}[{index}] must be a Tensor, but got {type(tensor)}."
            )
    return value


def _normalize_optional_tensor_list(value, name):
    if value is None:
        return None
    return _normalize_tensor_list(value, name)


def _normalize_bias(bias):
    if bias is None:
        return []
    return _normalize_tensor_list(bias, "bias")


def _normalize_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_TO_GE_DTYPE:
            raise TypeError(f"Unsupported dtype attr: {dtype}.")
        return _TORCH_DTYPE_TO_GE_DTYPE[dtype]
    if isinstance(dtype, int):
        return dtype - ACL_DTYPE_OFFSET if dtype >= ACL_DTYPE_OFFSET else dtype
    raise TypeError(f"Unsupported dtype attr type: {type(dtype)}.")


def _normalize_wrapper_dtype(dtype):
    dtype = _normalize_dtype(dtype)
    return None if dtype is None else dtype + ACL_DTYPE_OFFSET


def _normalize_v3_quant_dtype(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype is not None and dtype != GE_DTYPE_FLOAT8_E4M3FN:
        raise TypeError("swiglu_mode=2 only supports torch.float8_e4m3fn output.")
    return dtype


def _infer_v3_n(weight_scale):
    if weight_scale.dim() != WEIGHT_SCALE_RANK:
        raise ValueError("weight_scale[0] must be 4D in the V3 MXFP8 scenario.")
    return weight_scale.shape[WEIGHT_SCALE_N_AXIS]


class GroupedMatmulSwigluQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("grouped_matmul_swiglu_quant", category="gmm")

    def sources(self):
        return ["csrc/gmm/grouped_matmul_swiglu_quant.cpp"]

    def schema(self):
        return (
            "grouped_matmul_swiglu_quant("
            "Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale, Tensor group_list, "
            "*, Tensor? smooth_scale=None, Tensor[]? weight_assist_matrix=None, Tensor[]? bias=None, "
            f"int? dequant_mode={MX_QUANT_MODE}, int? dequant_dtype={TORCH_DTYPE_FLOAT}, "
            f"int? quant_mode={MX_QUANT_MODE}, int? quant_dtype={GE_DTYPE_FLOAT8_E4M3FN}, "
            "int? group_list_type=0, int[]? tuning_config=None, int? x_dtype=None, int? weight_dtype=None, "
            "int? weight_scale_dtype=None, int? x_scale_dtype=None, int? swiglu_mode=None, "
            "float? clamp_limit=None, float? glu_alpha=None, float? glu_bias=None, "
            'str round_mode="rint", int scale_alg=0, float dst_type_max=0.0) -> (Tensor, Tensor)'
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_matmul_swiglu_quant_meta(
            x,
            weight,
            weight_scale,
            x_scale,
            group_list,
            smooth_scale=None,
            weight_assist_matrix=None,
            bias=None,
            dequant_mode=MX_QUANT_MODE,
            dequant_dtype=TORCH_DTYPE_FLOAT,
            quant_mode=MX_QUANT_MODE,
            quant_dtype=GE_DTYPE_FLOAT8_E4M3FN,
            group_list_type=0,
            tuning_config=None,
            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=None,
            x_scale_dtype=None,
            swiglu_mode=None,
            clamp_limit=None,
            glu_alpha=None,
            glu_bias=None,
            round_mode="rint",
            scale_alg=0,
            dst_type_max=0.0,
        ):
            if swiglu_mode is None:
                raise ValueError(
                    "The cann_ops_transformer wrapper currently requires swiglu_mode=2; "
                    "legacy V2 calls remain available through torch_npu."
                )
            # Meta only checks the bridge inputs needed for output inference.
            # ACLNN validates supported modes, tensor counts and input dtypes.
            if not weight or not weight_scale:
                raise ValueError(
                    "weight and weight_scale must not be empty for output inference."
                )
            if x.dim() == 0:
                raise ValueError("x must have a dimension for output shape inference.")
            _normalize_v3_quant_dtype(quant_dtype)
            n = _infer_v3_n(weight_scale[0])
            if n <= 0 or n % SWIGLU_SPLIT_FACTOR != 0:
                raise ValueError("The logical N dimension must be positive and even.")
            m = x.shape[0]
            return (
                torch.empty(
                    (m, n // SWIGLU_SPLIT_FACTOR),
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                ),
                torch.empty(
                    (
                        m,
                        (
                            (n // SWIGLU_SPLIT_FACTOR)
                            + MX_GROUP_SIZE
                            - CEIL_DIV_ADJUSTMENT
                        )
                        // MX_GROUP_SIZE,
                        MX_SCALE_PAIR_SIZE,
                    ),
                    dtype=getattr(torch, "float8_e8m0fnu", torch.uint8),
                    device="meta",
                ),
            )


_op_builder = GroupedMatmulSwigluQuantOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _grouped_matmul_swiglu_quant(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=MX_QUANT_MODE,
    dequant_dtype=TORCH_DTYPE_FLOAT,
    quant_mode=MX_QUANT_MODE,
    quant_dtype=GE_DTYPE_FLOAT8_E4M3FN,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
    swiglu_mode=None,
    clamp_limit=None,
    glu_alpha=None,
    glu_bias=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
):
    module = _op_builder.load()
    return module.grouped_matmul_swiglu_quant(
        x,
        weight,
        weight_scale,
        x_scale,
        group_list,
        smooth_scale,
        weight_assist_matrix,
        bias,
        dequant_mode,
        dequant_dtype,
        quant_mode,
        quant_dtype,
        group_list_type,
        tuning_config,
        x_dtype,
        weight_dtype,
        weight_scale_dtype,
        x_scale_dtype,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
        round_mode,
        scale_alg,
        dst_type_max,
    )


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    *,
    smooth_scale: Optional[torch.Tensor] = None,
    weight_assist_matrix: Optional[List[torch.Tensor]] = None,
    bias: Optional[List[torch.Tensor]] = None,
    dequant_mode: Optional[int] = MX_QUANT_MODE,
    dequant_dtype: Optional[Union[int, torch.dtype]] = TORCH_DTYPE_FLOAT,
    quant_mode: Optional[int] = MX_QUANT_MODE,
    quant_dtype: Optional[Union[int, torch.dtype]] = torch.float8_e4m3fn,
    group_list_type: Optional[int] = 0,
    tuning_config: Optional[List[int]] = None,
    x_dtype: Optional[Union[int, torch.dtype]] = None,
    weight_dtype: Optional[Union[int, torch.dtype]] = None,
    weight_scale_dtype: Optional[Union[int, torch.dtype]] = FLOAT8_E8M0_DTYPE,
    x_scale_dtype: Optional[Union[int, torch.dtype]] = FLOAT8_E8M0_DTYPE,
    swiglu_mode: Optional[int] = None,
    clamp_limit: Optional[float] = None,
    glu_alpha: Optional[float] = None,
    glu_bias: Optional[float] = None,
    round_mode: str = "rint",
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """调用 GroupedMatmulSwigluQuantWeightNzV3。

    当前扩展仅支持 ``swiglu_mode=2`` 的 Ascend 950 单 Tensor MXFP8
    FRACTAL_NZ weight 场景；未传 ``swiglu_mode`` 的旧 V2 场景仍使用
    ``torch_npu.npu_grouped_matmul_swiglu_quant_v2``。
    """
    weight = _normalize_tensor_list(weight, "weight")
    weight_scale = _normalize_tensor_list(weight_scale, "weight_scale")
    weight_assist_matrix = _normalize_optional_tensor_list(
        weight_assist_matrix, "weight_assist_matrix"
    )
    bias = _normalize_bias(bias)
    if swiglu_mode is None:
        raise ValueError(
            "cann_ops_transformer.grouped_matmul_swiglu_quant currently requires swiglu_mode=2; "
            "use torch_npu.npu_grouped_matmul_swiglu_quant_v2 for legacy V2 scenarios."
        )
    # Normalize Python/dispatcher representations here. The C++ bridge guards
    # descriptor construction/output allocation; ACLNN owns scenario checks.
    return torch.ops.cann_ops_transformer.grouped_matmul_swiglu_quant(
        x,
        weight,
        weight_scale,
        x_scale,
        group_list,
        smooth_scale=smooth_scale,
        weight_assist_matrix=weight_assist_matrix,
        bias=bias,
        dequant_mode=dequant_mode,
        dequant_dtype=_normalize_dtype(dequant_dtype),
        quant_mode=quant_mode,
        quant_dtype=_normalize_dtype(quant_dtype),
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        x_dtype=_normalize_wrapper_dtype(x_dtype),
        weight_dtype=_normalize_wrapper_dtype(weight_dtype),
        weight_scale_dtype=_normalize_wrapper_dtype(weight_scale_dtype),
        x_scale_dtype=_normalize_wrapper_dtype(x_scale_dtype),
        swiglu_mode=swiglu_mode,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
