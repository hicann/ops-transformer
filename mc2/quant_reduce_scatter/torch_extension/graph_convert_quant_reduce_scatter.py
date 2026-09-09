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
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import (
        Tensor,
        TensorSpec,
        DataType,
        TensorType,
        torch_dtype_value_to_ge_type,
        torch_dtype_value_to_ge_proto_type,
        _ge_dtype_to_ge_proto_dtype,
        _ge_proto_dtype_to_ge_dtype,
    )
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    from torchair._ge_concrete_graph.ge_converter.converter_utils import *

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

from .quant_reduce_scatter import *
import logging

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
    def QuantReduceScatter(
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
            op_type="QuantReduceScatter",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("QuantReduceScatter")
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

    def _debug_print_ge_op_attrs():
        _graph = get_default_ge_graph()
        for _op in _graph.op:
            if _op.type == "QuantReduceScatter":
                _ws = (
                    _op.attr["world_size"].i
                    if "world_size" in _op.attr
                    else "NOT_FOUND"
                )
                _hbs = (
                    _op.attr["hccl_buffer_size"].i
                    if "hccl_buffer_size" in _op.attr
                    else "NOT_FOUND"
                )
                logging.info(
                    f"[DEBUG-GE-OP] OpName={_op.name}, world_size_attr={_ws}, hccl_buffer_size_attr={_hbs}"
                )
                for _k, _v in _op.attr.items():
                    logging.info(
                        f"[DEBUG-GE-OP] attr {_k}: has_i={_v.HasField('i')}, "
                        f"i={_v.i if _v.HasField('i') else 'N/A'}"
                    )
                for _idx, _inp in enumerate(_op.input_desc):
                    logging.info(
                        f"[DEBUG-GE-OP] input[{_idx}] name={_inp.name}, dtype={_inp.dtype}, "
                        f"shape={list(_inp.shape.dim)}, layout={_inp.layout}"
                    )
                logging.info(f"[DEBUG-GE-OP] input list: {list(_op.input)}")

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.npu_quant_reduce_scatter.default
    )
    def convert_npu_quant_reduce_scatter(
        context: Tensor,
        x: Tensor,
        scales: Tensor,
        hccl_buffer_size: int,
        world_size: int,
        reduce_op: Optional[str] = "sum",
        output_dtype: Optional[int] = None,
        x_dtype: Optional[int] = None,
        scales_dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        # 补充SymInt处理，添加hasattr(x, 'node')的校验，
        # 支持graph_type=2动态shape模式下hccl_buffer_size/world_size可能是SymInt的情况
        if hasattr(hccl_buffer_size, "node"):
            hccl_buffer_size = int(hccl_buffer_size.node)
        if hasattr(world_size, "node"):
            world_size = int(world_size.node)

        # 图模式下context张量被捕获为图输入占位符(arg)，编译期tiling读nPerServer读到野指针，
        # 改用wrapper缓存的CommContext int32数据构建ge.Const注入，保证tiling/运行期数据可用
        cached_ctx = get_cached_context_data(world_size, hccl_buffer_size)
        if cached_ctx is None:
            # 缓存未命中时context仍是图占位符，tiling会读到野指针，引发不可定位的kernel卡死，此处直接让其快速失败
            # 经wrapper调用时缓存必定已在其trace前写入，正常流程不会走到这里
            raise RuntimeError(
                "QuantReduceScatter graph mode: CommContext cache miss "
                f"(world_size={world_size}, hccl_buffer_size={hccl_buffer_size}). "
                "The context tensor would be captured as a graph placeholder and tiling "
                "would read a wild pointer. Please invoke the op through the "
                "cann_ops_transformer.ops.quant_reduce_scatter() wrapper so the "
                "CommContext cache is populated before graph tracing."
            )
        context = ge.Const(cached_ctx, dtype=int(DataType.DT_INT32))

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
        result = QuantReduceScatter(
            context=context,
            x=x,
            scales=scales,
            hccl_buffer_size=hccl_buffer_size,
            world_size=world_size,
            reduce_op=reduce_op,
            output_dtype=output_dtype,
        )
        _debug_print_ge_op_attrs()
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

    def convert_npu_quant_reduce_scatter(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
