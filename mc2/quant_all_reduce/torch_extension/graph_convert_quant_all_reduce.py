# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torchair
    from typing import Optional
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import (
        Tensor,
        TensorSpec,
        DataType,
        torch_dtype_value_to_ge_type,
        torch_dtype_value_to_ge_proto_type,
        _ge_proto_dtype_to_ge_dtype,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    from torchair._ge_concrete_graph.ge_converter.converter_utils import *

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

# 触发算子 schema/Meta/PrivateUse1 注册，确保 converter 注册时 torch.ops 符号已存在
from .quant_all_reduce import *
from ..common import QuantMteContextManager

if _TORCHAIR_AVAILABLE:
    # x valid dtype list
    X_DTYPE_SUPPORT_LIST = {
        DataType.DT_INT8,
        DataType.DT_HIFLOAT8,
        DataType.DT_FLOAT8_E5M2,
        DataType.DT_FLOAT8_E4M3FN,
    }
    # scales valid dtype list
    SCALES_DTYPE_SUPPORT_LIST = {DataType.DT_FLOAT, DataType.DT_FLOAT8_E8M0}

    @auto_convert_to_tensor([False, False, False], [False, False, False])
    def QuantAllReduce(
        context: Tensor,
        x: Tensor,
        scales: Tensor,
        *,
        hccl_buffer_size: int,
        world_size: int,
        reduce_op: str = "sum",
        output_dtype: int = 27,
        dependencies=[],
        node_name=None,
    ):
        # process inputs
        inputs = {
            "context": context,
            "x": x,
            "scales": scales,
        }

        # process attrs
        attrs = {
            "hccl_buffer_size": attr.Int(hccl_buffer_size),
            "reduce_op": attr.Str(reduce_op),
            "output_dtype": attr.Int(output_dtype),
            "world_size": attr.Int(world_size),
        }

        # process outputs
        outputs = [
            "out_put",
        ]

        return ge_op(
            op_type="QuantAllReduce",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("QuantAllReduce")
            .input("context", "DT_INT32")
            .input(
                "x",
                "DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1",
            )
            .input("scales", "DT_FLOAT, DT_FLOAT8_E8M0")
            .required_attr("hccl_buffer_size", attr.Int)
            .attr("reduce_op", attr.Str("sum"))
            .attr("output_dtype", attr.Int(27))
            .required_attr("world_size", attr.Int)
            .output("out_put", "DT_FLOAT16, DT_BF16, DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.npu_quant_all_reduce.default
    )
    def convert_npu_quant_all_reduce(
        context: Tensor,
        x: Tensor,
        scales: Tensor,
        hccl_buffer_size: int,
        world_size: int,
        hcom: str,
        reduce_op: Optional[str] = "sum",
        output_dtype: Optional[int] = None,
        x_dtype: Optional[int] = None,
        scales_dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        # graph_type=2 动态 shape 模式下 world_size 仍可能是 SymInt；
        # hccl_buffer_size 来自环境内置 buffer，与 shape 无关。
        if hasattr(world_size, "node"):
            world_size = int(world_size.node)

        # trace 阶段 context 是占位 tensor、hccl_buffer_size 是 0；
        # converter 在 GE 序列化/tiling 前创建幂等 context，并固化真实查询值。
        # 注意：converter 运行于 AOT 编译期的 FakeTensorMode 下，必须使用不做
        # tensor op 的 data 接口，直接以 host int32 列表构造 ge.Const。
        ctx_data, hccl_buffer_size = QuantMteContextManager(hcom).get_context_data()
        context = ge.Const(ctx_data, dtype=int(DataType.DT_INT32))

        if x_dtype is not None:
            if x_dtype == 296 or x_dtype == 297:
                const_x = ge.Const([1] * (x.rank - 1) + [2])
                shape_x = ge.Mul(ge.Shape(x), const_x)
                x = ge.Reshape(
                    ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype)), shape_x
                )
            else:
                x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
            x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
        if scales_dtype is not None:
            scales = ge.Bitcast(scales, type=torch_dtype_value_to_ge_type(scales_dtype))
            scales.desc.dtype = torch_dtype_value_to_ge_proto_type(scales_dtype)
        if output_dtype is not None:
            output_dtype = torch_dtype_value_to_ge_type(output_dtype)
        else:
            # default value is bfloat16
            output_dtype = DataType.DT_BF16
        check_dtype(x, scales)
        result = QuantAllReduce(
            context=context,
            x=x,
            scales=scales,
            hccl_buffer_size=hccl_buffer_size,
            world_size=world_size,
            reduce_op=reduce_op,
            output_dtype=output_dtype,
        )
        return result

    def check_dtype(x: Tensor, scales: Tensor):
        # Bitcast输出的Tensor的.dtype(_ge_dtype)仍为DT_UNDEFINED(28)，无法反映Bitcast后的真实类型
        # 真实类型由显示设置的desc.dtype(proto体系)记录，故迁移过来的校验改为desc.dtype并转DataType
        x_dtype = _ge_proto_dtype_to_ge_dtype(x.desc.dtype)
        scales_dtype = _ge_proto_dtype_to_ge_dtype(scales.desc.dtype)
        if x_dtype not in X_DTYPE_SUPPORT_LIST:
            raise AssertionError(
                f"The valid x dtype are: int8/hifloat8/float8_e5m2/float8_e4m3fn, "
                f"but input value is: {x_dtype}"
            )
        if scales_dtype not in SCALES_DTYPE_SUPPORT_LIST:
            raise AssertionError(
                f"The valid scales dtype are: float32/float8_e8m0, "
                f"but input value is: {scales_dtype}"
            )
        if (x_dtype == DataType.DT_INT8 or x_dtype == DataType.DT_HIFLOAT8) and (
            scales_dtype == DataType.DT_FLOAT8_E8M0
        ):
            raise AssertionError(
                "When x dtype is int8/hifloat8, scales dtype cannot be float8_e8m0"
            )

else:

    def convert_npu_quant_all_reduce(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
