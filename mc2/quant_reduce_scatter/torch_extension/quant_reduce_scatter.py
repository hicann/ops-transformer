# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from typing import Optional
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import QUANT_MTE_CONTEXT_ELEM_NUM
from ..common import QuantMteContextManager


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class QuantReduceScatterOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantReduceScatterOpBuilder, self).__init__(
            "npu_quant_reduce_scatter", category="mc2"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mc2/quant_reduce_scatter.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "npu_quant_reduce_scatter(Tensor context, Tensor x, Tensor scales, int hccl_buffer_size, "
            "int world_size, *, str hcom, str? reduce_op='sum', int? output_dtype=None, "
            "int? x_dtype=None, int? scales_dtype=None) -> Tensor"
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def npu_quant_reduce_scatter_meta(
            context,
            x,
            scales,
            hccl_buffer_size,
            world_size,
            hcom,
            reduce_op="sum",
            output_dtype=None,
            x_dtype=None,
            scales_dtype=None,
        ):
            torch._check(
                x is not None,
                lambda: "x cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )
            torch._check(
                scales is not None,
                lambda: "scales cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )

            world_size_list = [2, 4, 8]
            torch._check(
                world_size in world_size_list,
                lambda: "world_size must be in "
                + str(world_size_list)
                + ", but actual value is: "
                + str(world_size)
                + ops_error(ErrCode.VALUE),
            )

            if x.dim() == 2:
                size = [x.size(0) // world_size, x.size(1)]
            if x.dim() == 3:
                size = [x.size(0) * x.size(1) // world_size, x.size(2)]

            dtype = x.dtype
            if output_dtype is not None:
                dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype]
            else:
                dtype = torch.bfloat16

            return torch.empty(size, dtype=dtype, device="meta")


# Instantiate the builder
quant_reduce_scatter_op_builder = QuantReduceScatterOpBuilder()
quant_reduce_scatter_op_builder._ensure_initialized()


@impl(get_as_library(), quant_reduce_scatter_op_builder.name, "PrivateUse1")
def npu_quant_reduce_scatter(
    context,
    x,
    scales,
    hccl_buffer_size,
    world_size,
    hcom,
    reduce_op="sum",
    output_dtype=None,
    x_dtype=None,
    scales_dtype=None,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    # Compiles/loads the .so file
    op_module = quant_reduce_scatter_op_builder.load()
    return op_module.npu_quant_reduce_scatter(
        context,
        x,
        scales,
        hccl_buffer_size,
        world_size,
        reduce_op,
        output_dtype,
        x_dtype,
        scales_dtype,
    )


def quant_reduce_scatter(
    x: torch.Tensor,
    scales: torch.Tensor,
    hcom: str,
    world_size: int,
    *,
    reduce_op: Optional[str] = "sum",
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
):
    if isinstance(hcom, str):
        group_name = hcom
    else:
        group = torch.distributed.distributed_c10d._get_default_group()
        group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=False
        )

    if torch.compiler.is_compiling():
        # Dynamo trace 阶段只保留算子节点；converter 会创建真实 context 并覆写
        # hccl_buffer_size，因此这里使用无业务语义的占位数据，占位元素数与真实
        # context 一致（QUANT_MTE_CONTEXT_ELEM_NUM 的定义处有布局来源说明）。
        context = torch.zeros(
            QUANT_MTE_CONTEXT_ELEM_NUM,
            dtype=torch.int32,
            device=x.device,
        )
        hccl_buffer_size = 0
    else:
        context, hccl_buffer_size = QuantMteContextManager(group_name).get_context()

    return torch.ops.cann_ops_transformer.npu_quant_reduce_scatter(
        context,
        x,
        scales,
        hccl_buffer_size,
        world_size,
        hcom=group_name,
        reduce_op=reduce_op,
        output_dtype=output_dtype,
        x_dtype=x_dtype,
        scales_dtype=scales_dtype,
    )
