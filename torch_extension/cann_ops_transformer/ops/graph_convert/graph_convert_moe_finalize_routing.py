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


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False, False, False],
        [False, False, True, True, True, True, True, True, True, True, True],
    )
    def MoeFinalizeRouting(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        x: Optional[Tensor],
        alpha1: Optional[Tensor],
        alpha2: Optional[Tensor],
        v: Optional[Tensor],
        *,
        drop_pad_mode: int = 0,
        zero_expert_range: Optional[List[int]] = None,
        copy_expert_range: Optional[List[int]] = None,
        constant_expert_range: Optional[List[int]] = None,
        k: int = 1,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeFinalizeRoutingV2)\n
        .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .INPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(expert_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(alpha1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(alpha2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(zero_expert_range, ListInt, {})\n
        .ATTR(copy_expert_range, ListInt, {})\n
        .ATTR(constant_expert_range, ListInt, {})\n
        .ATTR(k, Int, 1)\n
        """
        if zero_expert_range is None:
            zero_expert_range = []
        if copy_expert_range is None:
            copy_expert_range = []
        if constant_expert_range is None:
            constant_expert_range = []
        if dependencies is None:
            dependencies = []

        inputs = {
            "expanded_x": expanded_x,
            "expanded_row_idx": expanded_row_idx,
            "x1": x1,
            "x2": x2,
            "bias": bias,
            "scales": scales,
            "expert_idx": expert_idx,
            "x": x,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "v": v,
        }

        attrs = {
            "drop_pad_mode": attr.Int(drop_pad_mode),
            "zero_expert_range": attr.ListInt(zero_expert_range),
            "copy_expert_range": attr.ListInt(copy_expert_range),
            "constant_expert_range": attr.ListInt(constant_expert_range),
            "k": attr.Int(k),
        }

        outputs = ["y"]

        return ge_op(
            op_type="MoeFinalizeRoutingV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MoeFinalizeRoutingV2")
            .input("expanded_x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .input("expanded_row_idx", "DT_INT32")
            .optional_input("x1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("x2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("bias", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("scales", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("expert_idx", "DT_INT32")
            .optional_input("x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("alpha1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("alpha2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("v", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .attr("drop_pad_mode", attr.Int(0))
            .attr("zero_expert_range", attr.ListInt([]))
            .attr("copy_expert_range", attr.ListInt([]))
            .attr("constant_expert_range", attr.ListInt([]))
            .attr("k", attr.Int(1))
            .output("y", "DT_FLOAT, DT_FLOAT16, DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_finalize_routing.default
    )
    def convert_moe_finalize_routing(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        x: Optional[Tensor],
        alpha1: Optional[Tensor],
        alpha2: Optional[Tensor],
        v: Optional[Tensor],
        drop_pad_mode: int = 0,
        zero_expert_range: Optional[List[int]] = None,
        copy_expert_range: Optional[List[int]] = None,
        constant_expert_range: Optional[List[int]] = None,
        k: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        return MoeFinalizeRouting(
            expanded_x=expanded_x,
            expanded_row_idx=expanded_row_idx,
            x1=x1,
            x2=x2,
            bias=bias,
            scales=scales,
            expert_idx=expert_idx,
            x=x,
            alpha1=alpha1,
            alpha2=alpha2,
            v=v,
            drop_pad_mode=drop_pad_mode,
            zero_expert_range=zero_expert_range,
            copy_expert_range=copy_expert_range,
            constant_expert_range=constant_expert_range,
            k=k,
        )
