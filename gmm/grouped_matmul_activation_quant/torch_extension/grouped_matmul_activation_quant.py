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
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


GE_DTYPE_FLOAT = 0
GE_DTYPE_FLOAT8_E5M2 = 35
GE_DTYPE_FLOAT8_E4M3FN = 36
GE_DTYPE_FLOAT8_E8M0 = 37
GE_DTYPE_FLOAT4_E2M1 = 40
GE_DTYPE_FLOAT4_E1M2 = 41
ACL_DTYPE_OFFSET = 256
DEFAULT_Y_DTYPE = GE_DTYPE_FLOAT8_E4M3FN

FLOAT8_E8M0_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", torch.uint8)
)

_TORCH_DTYPE_TO_GE_DTYPE = {
    torch.float32: GE_DTYPE_FLOAT,
    torch.float8_e5m2: GE_DTYPE_FLOAT8_E5M2,
    torch.float8_e4m3fn: GE_DTYPE_FLOAT8_E4M3FN,
    FLOAT8_E8M0_DTYPE: GE_DTYPE_FLOAT8_E8M0,
}

_GE_DTYPE_TO_TORCH_DTYPE = {
    GE_DTYPE_FLOAT8_E5M2: torch.float8_e5m2,
    GE_DTYPE_FLOAT8_E4M3FN: torch.float8_e4m3fn,
    GE_DTYPE_FLOAT8_E8M0: FLOAT8_E8M0_DTYPE,
}

_GE_DTYPE_TO_NAME = {
    GE_DTYPE_FLOAT: "FLOAT",
    GE_DTYPE_FLOAT8_E5M2: "FLOAT8_E5M2",
    GE_DTYPE_FLOAT8_E4M3FN: "FLOAT8_E4M3FN",
    GE_DTYPE_FLOAT8_E8M0: "FLOAT8_E8M0",
    GE_DTYPE_FLOAT4_E2M1: "FLOAT4_E2M1",
    GE_DTYPE_FLOAT4_E1M2: "FLOAT4_E1M2",
}


def _normalize_tensor_list(value, name):
    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a TensorList (list of Tensor), but got a single Tensor."
        )
    if isinstance(value, list):
        for index, tensor in enumerate(value):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{name}[{index}] must be a Tensor, but got {type(tensor)}."
                )
        return value
    raise TypeError(
        f"{name} must be a TensorList (list of Tensor), but got {type(value)}."
    )


def _normalize_bias(bias):
    if bias is None:
        return []
    return _normalize_tensor_list(bias, "bias")


def _normalize_attr_dtype(dtype, default=None):
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_TO_GE_DTYPE:
            raise TypeError(f"Unsupported dtype attr: {dtype}.")
        return _TORCH_DTYPE_TO_GE_DTYPE[dtype]
    if isinstance(dtype, int):
        if dtype >= ACL_DTYPE_OFFSET:
            return dtype - ACL_DTYPE_OFFSET
        return dtype
    raise TypeError(f"Unsupported dtype attr type: {type(dtype)}.")


def _normalize_wrapper_dtype(dtype):
    if dtype is None:
        return None
    ge_dtype = _normalize_attr_dtype(dtype)
    if ge_dtype in (
        GE_DTYPE_FLOAT,
        GE_DTYPE_FLOAT8_E5M2,
        GE_DTYPE_FLOAT8_E4M3FN,
        GE_DTYPE_FLOAT8_E8M0,
    ):
        return ge_dtype + ACL_DTYPE_OFFSET
    return dtype


def _to_torch_dtype(dtype):
    ge_dtype = _normalize_attr_dtype(dtype)
    if ge_dtype not in _GE_DTYPE_TO_TORCH_DTYPE:
        dtype_name = _GE_DTYPE_TO_NAME.get(ge_dtype, "UNKNOWN")
        raise TypeError(f"Unsupported y_dtype: {dtype_name}.")
    return _GE_DTYPE_TO_TORCH_DTYPE[ge_dtype]


def _get_effective_x_ge_dtype(x, x_dtype):
    if x_dtype is not None:
        return _normalize_attr_dtype(x_dtype)
    return _TORCH_DTYPE_TO_GE_DTYPE.get(x.dtype)


def _resolve_y_dtype(y_dtype, x, x_dtype):
    if y_dtype is None:
        x_ge_dtype = _get_effective_x_ge_dtype(x, x_dtype)
        if x_ge_dtype in (GE_DTYPE_FLOAT8_E4M3FN, GE_DTYPE_FLOAT8_E5M2):
            return x_ge_dtype
        raise TypeError("y_dtype is None only supports inferring from FP8 x dtype.")
    return _normalize_attr_dtype(y_dtype)


def _infer_nz_logical_n(weight_scale):
    # Runtime callers normalize MX weight to the non-transposed logical layout before this op.
    # Thus 4D MX weight_scale is [E, ceil(K / 64), N, 2], and logical N is dim2.
    return weight_scale.shape[2]


class GroupedMatmulActivationQuantOpBuilder(OpBuilder):
    def __init__(self):
        super(GroupedMatmulActivationQuantOpBuilder, self).__init__(
            "grouped_matmul_activation_quant", category="gmm"
        )

    def sources(self):
        return ["csrc/gmm/grouped_matmul_activation_quant.cpp"]

    def schema(self) -> str:
        return (
            "grouped_matmul_activation_quant("
            "Tensor x, Tensor group_list, Tensor[] weight, Tensor[] weight_scale, str activation_type, "
            "*, Tensor[]? bias=None, Tensor? x_scale=None, "
            "int group_list_type=0, int[]? tuning_config=None, "
            'str? quant_mode=None, int? y_dtype=None, str round_mode="rint", int scale_alg=0, '
            "float dst_type_max=0.0, int? x_dtype=None, int? weight_dtype=None, "
            "int? weight_scale_dtype=None, int? x_scale_dtype=None) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_matmul_activation_quant_meta(
            x,
            group_list,
            weight,
            weight_scale,
            activation_type,
            bias=None,
            x_scale=None,
            group_list_type=0,
            tuning_config=None,
            quant_mode=None,
            y_dtype=None,
            round_mode="rint",
            scale_alg=0,
            dst_type_max=0.0,
            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=None,
            x_scale_dtype=None,
        ):
            if len(weight) == 0 or weight[0] is None:
                raise ValueError(
                    "weight must contain at least one non-null tensor for meta output inference."
                )
            if len(weight_scale) == 0 or weight_scale[0] is None:
                raise ValueError(
                    "weight_scale must contain at least one non-null tensor for meta output inference."
                )
            if x_scale is None:
                raise ValueError("x_scale must be provided for meta output inference.")
            if x.dim() <= 0:
                raise ValueError(
                    "x must have at least one dimension for meta output inference."
                )
            if weight_scale[0].dim() <= 2:
                raise ValueError(
                    "weight_scale must have at least 3 dimensions for meta output inference."
                )

            m = x.shape[0]
            n = _infer_nz_logical_n(weight_scale[0])

            y_dtype_value = _to_torch_dtype(_resolve_y_dtype(y_dtype, x, x_dtype))
            y = torch.empty((m, n), dtype=y_dtype_value, device="meta")
            y_scale = torch.empty(
                (m, math.ceil(n / 64), 2), dtype=FLOAT8_E8M0_DTYPE, device="meta"
            )
            return (y, y_scale)


_grouped_matmul_activation_quant_op_builder = GroupedMatmulActivationQuantOpBuilder()
_grouped_matmul_activation_quant_op_builder._ensure_initialized()


@impl(get_as_library(), _grouped_matmul_activation_quant_op_builder.name, "PrivateUse1")
def _grouped_matmul_activation_quant(
    x,
    group_list,
    weight,
    weight_scale,
    activation_type,
    bias=None,
    x_scale=None,
    group_list_type=0,
    tuning_config=None,
    quant_mode=None,
    y_dtype=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
):
    _op_module = _grouped_matmul_activation_quant_op_builder.load()
    return _op_module.grouped_matmul_activation_quant(
        x,
        group_list,
        weight,
        weight_scale,
        activation_type,
        bias,
        x_scale,
        group_list_type,
        tuning_config,
        quant_mode,
        y_dtype,
        round_mode,
        scale_alg,
        dst_type_max,
        x_dtype,
        weight_dtype,
        weight_scale_dtype,
        x_scale_dtype,
    )


def grouped_matmul_activation_quant(
    x: torch.Tensor,
    group_list: torch.Tensor,
    weight: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    activation_type: str,
    *,
    bias: Optional[List[torch.Tensor]] = None,
    x_scale: Optional[torch.Tensor] = None,
    group_list_type: int = 0,
    tuning_config: Optional[List[int]] = None,
    quant_mode: Optional[str] = None,
    y_dtype: Optional[torch.dtype] = None,
    round_mode: str = "rint",
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
    x_dtype: Optional[int] = None,
    weight_dtype: Optional[int] = None,
    weight_scale_dtype: Optional[int] = None,
    x_scale_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GroupedMatmulActivationQuant torch接口，封装 aclnnGroupedMatmulActivationQuantWeightNz。

    Args:
        x (Tensor): 左矩阵，shape为 ``(M, K)``，dtype支持 ``torch.float8_e4m3fn`` 或
            ``torch.float8_e5m2``。
        group_list (Tensor): 分组信息，1D Tensor，dtype为 ``torch.int64``，第一维表示group数E，
            当前E取值范围为[1, 1024]。
        weight (List[Tensor]): 右矩阵dynamic input，当前MXFP8仅支持长度为1。
            元素为3D逻辑Tensor，MX WeightNZ运行时调用者会将其规整为非转置逻辑布局，
            传入前需要通过 ``torch_npu.npu_format_cast(weight, 29)`` 转为FRACTAL_NZ格式。
        weight_scale (List[Tensor]): weight的MX量化scale，当前MXFP8仅支持长度为1。
            shape为 ``(E, ceil(K / 64), N, 2)``，torch层根据第2维推导逻辑N。
        activation_type (str): 激活函数类型，当前仅支持 ``"gelu_tanh"``。
        bias (List[Tensor], optional): bias dynamic input。当前MXFP8仅支持传None、空TensorList或单个空Tensor。
        x_scale (Tensor, optional): x的MX量化scale。当前MXFP8必须传入，shape为
            ``(M, ceil(K / 64), 2)``。
        group_list_type (int): group_list语义类型，当前支持0或1。
        tuning_config (List[int], optional): 预留调优参数。
        quant_mode (str, optional): 量化模式，torch层不做解析，直接透传到aclnn层处理。
        y_dtype (torch.dtype, optional): 输出y的数据类型，支持 ``torch.float8_e4m3fn`` 和
            ``torch.float8_e5m2``；为None时默认推导为x的数据类型。
        round_mode (str): 舍入模式，当前仅支持 ``"rint"``。
        scale_alg (int): scale算法，当前支持0或1。
        dst_type_max (float): 预留参数，当前仅支持0.0。
        x_dtype (int, optional): x的dtype wrapper覆盖值，传入torch_npu dtype枚举。
        weight_dtype (int, optional): weight的dtype wrapper覆盖值，传入torch_npu dtype枚举。
        weight_scale_dtype (int, optional): weight_scale的dtype wrapper覆盖值，传入torch_npu dtype枚举。
        x_scale_dtype (int, optional): x_scale的dtype wrapper覆盖值，传入torch_npu dtype枚举。

    Returns:
        Tuple[Tensor, Tensor]: ``(y, y_scale)``，其中 ``y`` shape为 ``(M, N)``，
        ``y_scale`` shape为 ``(M, ceil(N / 64), 2)``。
    """
    weight = _normalize_tensor_list(weight, "weight")
    weight_scale = _normalize_tensor_list(weight_scale, "weight_scale")
    bias = _normalize_bias(bias)
    y_dtype = (
        None
        if y_dtype is None
        else _normalize_wrapper_dtype(y_dtype) - ACL_DTYPE_OFFSET
    )
    return torch.ops.cann_ops_transformer.grouped_matmul_activation_quant(
        x,
        group_list,
        weight,
        weight_scale,
        activation_type,
        bias=bias,
        x_scale=x_scale,
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        quant_mode=quant_mode,
        y_dtype=y_dtype,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
        x_dtype=x_dtype,
        weight_dtype=weight_dtype,
        weight_scale_dtype=weight_scale_dtype,
        x_scale_dtype=x_scale_dtype,
    )
