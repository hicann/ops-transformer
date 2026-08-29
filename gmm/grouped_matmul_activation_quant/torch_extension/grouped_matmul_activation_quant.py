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
from typing import List, Optional, Tuple, Union

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

FLOAT8_E8M0_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", torch.uint8)
)
FLOAT4_E2M1_DTYPE = getattr(
    torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None)
)
FLOAT4_E1M2_DTYPE = getattr(
    torch_npu, "float4_e1m2fn_x2", getattr(torch, "float4_e1m2fn_x2", None)
)

_TORCH_DTYPE_TO_GE_DTYPE = {
    torch.float32: GE_DTYPE_FLOAT,
    torch.float8_e5m2: GE_DTYPE_FLOAT8_E5M2,
    torch.float8_e4m3fn: GE_DTYPE_FLOAT8_E4M3FN,
    FLOAT8_E8M0_DTYPE: GE_DTYPE_FLOAT8_E8M0,
}
if FLOAT4_E2M1_DTYPE is not None:
    _TORCH_DTYPE_TO_GE_DTYPE[FLOAT4_E2M1_DTYPE] = GE_DTYPE_FLOAT4_E2M1
if FLOAT4_E1M2_DTYPE is not None:
    _TORCH_DTYPE_TO_GE_DTYPE[FLOAT4_E1M2_DTYPE] = GE_DTYPE_FLOAT4_E1M2

_GE_DTYPE_TO_TORCH_DTYPE = {
    GE_DTYPE_FLOAT8_E5M2: torch.float8_e5m2,
    GE_DTYPE_FLOAT8_E4M3FN: torch.float8_e4m3fn,
    GE_DTYPE_FLOAT8_E8M0: FLOAT8_E8M0_DTYPE,
    GE_DTYPE_FLOAT4_E2M1: torch.uint8,
    GE_DTYPE_FLOAT4_E1M2: torch.uint8,
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
        GE_DTYPE_FLOAT4_E2M1,
        GE_DTYPE_FLOAT4_E1M2,
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
        if x_ge_dtype in (GE_DTYPE_FLOAT4_E2M1, GE_DTYPE_FLOAT4_E1M2):
            return x_ge_dtype
        raise TypeError(
            "y_dtype is None only supports inferring from FP8 or FP4 x dtype."
        )
    return _normalize_attr_dtype(y_dtype)


def _infer_nz_logical_n(weight_scale):
    # Both the non-transposed scale and the view produced by
    # scale_source.transpose(-3, -2) are [E, ceil(K / 64), N, 2].
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

            y_ge_dtype = _resolve_y_dtype(y_dtype, x, x_dtype)
            y_dtype_value = _to_torch_dtype(y_ge_dtype)
            y_n = (
                math.ceil(n / 2)
                if y_ge_dtype in (GE_DTYPE_FLOAT4_E2M1, GE_DTYPE_FLOAT4_E1M2)
                else n
            )
            y = torch.empty((m, y_n), dtype=y_dtype_value, device="meta")
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
    y_dtype: Optional[Union[torch.dtype, int]] = None,
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
        x (Tensor): 左矩阵。MXFP8 shape为 ``(M, K)``，dtype支持 ``torch.float8_e4m3fn`` 或
            ``torch.float8_e5m2``；MXFP4使用 ``torch.uint8`` 存放打包数据，shape为 ``(M, K / 2)``，
            并通过 ``x_dtype`` 指定逻辑FLOAT4类型。
        group_list (Tensor): 分组信息，1D Tensor，dtype为 ``torch.int64``，第一维表示group数E，
            当前E取值范围为[1, 1024]。
        weight (List[Tensor]): 右矩阵dynamic input，当前MX仅支持长度为1。
            元素为3D逻辑Tensor；MXFP4使用 ``torch.uint8`` 存放打包数据并通过 ``weight_dtype``
            指定逻辑FLOAT4类型。转置场景应先将源Weight通过
            ``torch_npu.npu_format_cast(weight, 29)`` 转为FRACTAL_NZ格式，再对末两维执行transpose。
        weight_scale (List[Tensor]): weight的MX量化scale，当前MX仅支持长度为1。
            传入算子的shape为 ``(E, ceil(K / 64), N, 2)``；转置场景由源shape
            ``(E, N, ceil(K / 64), 2)`` 对中间两维执行transpose得到。
        activation_type (str): 激活函数类型，当前仅支持 ``"gelu_tanh"``。
        bias (List[Tensor], optional): bias dynamic input。当前MX仅支持传None、空TensorList或单个空Tensor。
        x_scale (Tensor, optional): x的MX量化scale。当前MX必须传入，shape为
            ``(M, ceil(K / 64), 2)``。
        group_list_type (int): group_list语义类型，当前支持0或1。
        tuning_config (List[int], optional): 预留调优参数。
        quant_mode (str, optional): 量化模式，torch层不做解析，直接透传到aclnn层处理。
        y_dtype (torch.dtype | int, optional): 输出y的数据类型，可传两种FP8/两种FP4的 ``torch.dtype``
            或对应GE dtype整数；为None时默认推导为x的逻辑数据类型。FP4输出以打包 ``torch.uint8`` 返回，
            末维物理长度为 ``ceil(N / 2)``。
        round_mode (str): 舍入模式，当前仅支持 ``"rint"``。
        scale_alg (int): scale算法，支持0、1、2；1仅支持FP8输出，2仅支持FLOAT4_E2M1输出。
        dst_type_max (float): ``scale_alg=2``时支持0.0或[6.0, 12.0]，其他算法仅支持0.0。
        x_dtype (int, optional): x的dtype wrapper覆盖值。MXFP4传入
            ``torch_npu.float4_e2m1fn_x2`` 或 ``torch_npu.float4_e1m2fn_x2``。
        weight_dtype (int, optional): weight的dtype wrapper覆盖值。MXFP4传入
            ``torch_npu.float4_e2m1fn_x2`` 或 ``torch_npu.float4_e1m2fn_x2``。
        weight_scale_dtype (int, optional): weight_scale的dtype wrapper覆盖值，传入torch_npu dtype枚举。
        x_scale_dtype (int, optional): x_scale的dtype wrapper覆盖值，传入torch_npu dtype枚举。

    Returns:
        Tuple[Tensor, Tensor]: ``(y, y_scale)``。FP8 ``y`` shape为 ``(M, N)``；FP4的Torch载体shape为
        ``(M, ceil(N / 2))``，其ACL/GE逻辑shape仍为 ``(M, N)``。``y_scale`` shape为
        ``(M, ceil(N / 64), 2)``。
    """
    weight = _normalize_tensor_list(weight, "weight")
    weight_scale = _normalize_tensor_list(weight_scale, "weight_scale")
    bias = _normalize_bias(bias)
    x_dtype = _normalize_wrapper_dtype(x_dtype)
    weight_dtype = _normalize_wrapper_dtype(weight_dtype)
    weight_scale_dtype = _normalize_wrapper_dtype(weight_scale_dtype)
    x_scale_dtype = _normalize_wrapper_dtype(x_scale_dtype)
    # y_dtype is an operator attribute (GE dtype), unlike the tensor wrapper
    # dtype overrides below.  Normalize it once and never subtract an ACL
    # wrapper offset from an already-normalized GE value.
    y_dtype = None if y_dtype is None else _normalize_attr_dtype(y_dtype)
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
