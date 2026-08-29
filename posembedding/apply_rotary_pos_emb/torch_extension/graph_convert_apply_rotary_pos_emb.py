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
    from typing import List, Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


_LAYOUT_MAP = {
    "BSH": 1,
    "BSND": 1,
    "SBND": 2,
    "BNSD": 3,
    "TND": 4,
}

_SUPPORTED_ROTARY_MODES = ("half", "quarter", "interleave")


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
    def apply_rotary_pos_emb(
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        layout: int = 1,
        rotary_mode: str = "half",
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(ApplyRotaryPosEmb)\n
        .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .OUTPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .OUTPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .ATTR(layout, Int, 1)\n
        .ATTR(rotary_mode, String, "half")\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "cos": cos,
            "sin": sin,
        }
        attrs = {
            "layout": attr.Int(layout),
            "rotary_mode": attr.Str(rotary_mode),
        }
        outputs = ["query", "key"]

        return ge_op(
            op_type="ApplyRotaryPosEmb",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("ApplyRotaryPosEmb")
            .input("query", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("key", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("cos", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("sin", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .attr("layout", attr.Int(1))
            .attr("rotary_mode", attr.Str("half"))
            .output("query", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .output("key", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.apply_rotary_pos_emb.default
    )
    def convert_apply_rotary_pos_emb(
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
        layout: str = "BSND",
        rotary_mode: str = "half",
        meta_outputs: TensorSpec = None,
    ):
        if layout not in _LAYOUT_MAP:
            raise NotImplementedError(
                f"layout only supports BSH/BSND/SBND/BNSD/TND, got '{layout}'."
            )
        if rotary_mode not in _SUPPORTED_ROTARY_MODES:
            raise NotImplementedError(
                f"rotary_mode only supports half/quarter/interleave, got '{rotary_mode}'."
            )
        return apply_rotary_pos_emb(
            query=query,
            key=key,
            cos=cos,
            sin=sin,
            layout=_LAYOUT_MAP[layout],
            rotary_mode=rotary_mode,
        )
else:

    def convert_apply_rotary_pos_emb(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
