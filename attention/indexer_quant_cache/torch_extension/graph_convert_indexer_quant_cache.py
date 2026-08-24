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

    @auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
    def IndexerQuantCache(
        cache: Tensor,
        cache_scale: Tensor,
        x: Tensor,
        slot_mapping: Tensor,
        *,
        quant_mode: int = 1,
        round_scale: bool = True,
        x_scale: float = 1.0,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(IndexerQuantCache)\n
        .INPUT(cache, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_UINT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .INPUT(cache_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))\n
        .INPUT(slot_mapping, TensorType({DT_INT32}))\n
        .OUTPUT(cache, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_UINT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .OUTPUT(cache_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .ATTR(quant_mode, Int, 1)\n
        .ATTR(round_scale, Bool, true)\n
        .ATTR(x_scale, Float, 1.0)\n
        """
        inputs = {
            "cache": cache,
            "cache_scale": cache_scale,
            "x": x,
            "slot_mapping": slot_mapping,
        }

        attrs = {
            "quant_mode": attr.Int(quant_mode),
            "round_scale": attr.Bool(round_scale),
            "x_scale": attr.Float(x_scale),
        }

        outputs = [
            "cache",
            "cache_scale",
        ]

        return ge_op(
            op_type="IndexerQuantCache",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("IndexerQuantCache")
            .input(
                "cache",
                "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_UINT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
            )
            .input("cache_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .input("x", "DT_FLOAT16, DT_BF16")
            .input("slot_mapping", "DT_INT32")
            .attr("quant_mode", attr.Int(1))
            .attr("round_scale", attr.Bool(True))
            .attr("x_scale", attr.Float(1.0))
            .output(
                "cache",
                "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_UINT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
            )
            .output("cache_scale", "DT_FLOAT, DT_FLOAT8_E8M0"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.indexer_quant_cache.default
    )
    def convert_indexer_quant_cache(
        cache: Tensor,
        cache_scale: Tensor,
        x: Tensor,
        slot_mapping: Tensor,
        quant_mode: str = "fp8",
        round_scale: bool = True,
        x_scale: float = 1.0,
        meta_outputs: TensorSpec = None,
    ):
        # 对外 str quant_mode → 算子侧 int（与 eager 路径共用同一映射）
        from .indexer_quant_cache import _resolve_quant_mode

        out = IndexerQuantCache(
            cache=cache,
            cache_scale=cache_scale,
            x=x,
            slot_mapping=slot_mapping,
            quant_mode=_resolve_quant_mode(quant_mode),
            round_scale=round_scale,
            x_scale=x_scale,
        )
        return out[0]
