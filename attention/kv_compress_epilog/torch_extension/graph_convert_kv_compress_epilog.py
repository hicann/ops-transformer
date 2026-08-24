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
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False], [False, False, False])
    def KvCompressEpilog(
        cache: Tensor,
        x: Tensor,
        slot_mapping: Tensor,
        *,
        quant_group_size: int = 64,
        quant_mode: int = 1,
        round_scale: bool = True,
        x_scale: float = 1.0,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(KvCompressEpilog)\n
        .INPUT(cache, TensorType({DT_UINT8}))\n
        .INPUT(x, TensorType({DT_BF16}))\n
        .INPUT(slot_mapping, TensorType({DT_INT32, DT_INT64}))\n
        .OUTPUT(cache, TensorType({DT_UINT8}))\n
        .ATTR(quant_group_size, Int, 64)\n
        .ATTR(quant_mode, Int, 1)\n
        .ATTR(round_scale, Bool, true)\n
        .ATTR(x_scale, Float, 1.0)\n
        """
        inputs = {
            "cache": cache,
            "x": x,
            "slot_mapping": slot_mapping,
        }

        attrs = {
            "quant_group_size": attr.Int(quant_group_size),
            "quant_mode": attr.Int(quant_mode),
            "round_scale": attr.Bool(round_scale),
            "x_scale": attr.Float(x_scale),
        }

        outputs = [
            "cache",
        ]

        return ge_op(
            op_type="KvCompressEpilog",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("KvCompressEpilog")
            .input("cache", "DT_UINT8")
            .input("x", "DT_BF16")
            .input("slot_mapping", "DT_INT32, DT_INT64")
            .attr("quant_group_size", attr.Int(64))
            .attr("quant_mode", attr.Int(1))
            .attr("round_scale", attr.Bool(True))
            .attr("x_scale", attr.Float(1.0))
            .output("cache", "DT_UINT8"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.kv_compress_epilog.default
    )
    def convert_kv_compress_epilog(
        cache: Tensor,
        x: Tensor,
        slot_mapping: Tensor,
        quant_group_size: int = 64,
        quant_mode: str = "fp8_e8m0",
        round_scale: bool = True,
        x_scale: float = 1.0,
        meta_outputs: TensorSpec = None,
    ):
        # 对外 str quant_mode → 算子侧 int（与 eager 路径共用同一映射）
        from .kv_compress_epilog import _resolve_quant_mode

        return KvCompressEpilog(
            cache=cache,
            x=x,
            slot_mapping=slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=_resolve_quant_mode(quant_mode),
            round_scale=round_scale,
            x_scale=x_scale,
        )
