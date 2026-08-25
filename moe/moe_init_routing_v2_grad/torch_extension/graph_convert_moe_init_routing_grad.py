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
        [False, False],
        [False, False],
    )
    def _moe_init_routing_grad(
        grad_expanded_x: Tensor,
        expanded_row_idx: Tensor,
        *,
        top_k: int,
        drop_pad_mode: int = 0,
        active_num: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeInitRoutingV2Grad)\n
        .INPUT(grad_expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .INPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OUTPUT(grad_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .REQUIRED_ATTR(top_k, Int)\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(active_num, Int, 0)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "grad_expanded_x": grad_expanded_x,
            "expanded_row_idx": expanded_row_idx,
        }

        attrs = {
            "top_k": attr.Int(top_k),
            "drop_pad_mode": attr.Int(drop_pad_mode),
            "active_num": attr.Int(active_num),
        }

        outputs = ["grad_x"]

        return ge_op(
            op_type="MoeInitRoutingV2Grad",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MoeInitRoutingV2Grad")
            .input("grad_expanded_x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .input("expanded_row_idx", "DT_INT32")
            .output("grad_x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .required_attr("top_k", attr.Int)
            .attr("drop_pad_mode", attr.Int(0))
            .attr("active_num", attr.Int(0)),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_init_routing_grad.default
    )
    def convert_moe_init_routing_grad(
        grad_expanded_x: Tensor,
        expanded_row_idx: Tensor,
        top_k: int,
        drop_pad_mode: int = 0,
        active_num: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        return _moe_init_routing_grad(
            grad_expanded_x=grad_expanded_x,
            expanded_row_idx=expanded_row_idx,
            top_k=top_k,
            drop_pad_mode=drop_pad_mode,
            active_num=active_num,
        )
