# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
try:
    import torch
    import torch_npu
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.utils import dtype_promote
    from torchair.ge._ge_graph import DataType, auto_convert_to_tensor
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge import attr
    from typing import Optional, Union

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False],
        [False, False, True, True, True, True],
    )
    def MoeInitRoutingV4(
        x: Tensor,
        expert_idx: Tensor,
        scale: Optional[Tensor],
        offset: Optional[Tensor],
        active_num: Optional[Tensor],
        topk_weight: Optional[Tensor],
        *,
        expert_capacity: int,
        expert_num: int,
        drop_pad_mode: int,
        expert_tokens_num_type: int,
        expert_tokens_num_flag: bool,
        quant_mode: int,
        active_expert_range: Optional[list],
        row_idx_type: int,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeInitRoutingV4)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8}))\n
        .INPUT(expert_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(active_num, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(topk_weight, TensorType({DT_FLOAT}))\n
        .OUTPUT(expanded_x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8}))\n
        .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))\n
        .OUTPUT(expanded_scale, TensorType({DT_FLOAT}))\n
        .OUTPUT(expanded_topk_weight, TensorType({DT_FLOAT}))\n
        .ATTR(expert_capacity, Int, -1)\n
        .ATTR(expert_num, Int, -1)\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(expert_tokens_num_type, Int, 0)\n
        .ATTR(expert_tokens_num_flag, Bool, False)\n
        .ATTR(quant_mode, Int, -1)\n
        .ATTR(active_expert_range, ListInt, {-1, -1})\n
        .ATTR(row_idx_type, Int, 0)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "expert_idx": expert_idx,
            "scale": scale,
            "offset": offset,
            "active_num": active_num,
            "topk_weight": topk_weight,
        }

        attrs = {
            "expert_capacity": attr.Int(expert_capacity),
            "expert_num": attr.Int(expert_num),
            "drop_pad_mode": attr.Int(drop_pad_mode),
            "expert_tokens_num_type": attr.Int(expert_tokens_num_type),
            "expert_tokens_num_flag": attr.Bool(expert_tokens_num_flag),
            "quant_mode": attr.Int(quant_mode),
            "row_idx_type": attr.Int(row_idx_type),
        }
        if active_expert_range is not None:
            attrs["active_expert_range"] = attr.ListInt(active_expert_range)
        else:
            attrs["active_expert_range"] = attr.ListInt([])

        outputs = [
            "expanded_x",
            "expanded_row_idx",
            "expert_tokens_count_or_cumsum",
            "expanded_scale",
            "expanded_topk_weight",
        ]

        return ge_op(
            op_type="MoeInitRoutingV4",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeInitRoutingV4")
            .input(
                "x",
                "DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_HIFLOAT8, "
                "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input("expert_idx", "DT_INT32")
            .optional_input("scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("offset", "DT_FLOAT")
            .optional_input("active_num", "DT_INT64")
            .optional_input("topk_weight", "DT_FLOAT")
            .attr("expert_capacity", attr.Int(-1))
            .attr("expert_num", attr.Int(-1))
            .attr("drop_pad_mode", attr.Int(0))
            .attr("expert_tokens_num_type", attr.Int(0))
            .attr("expert_tokens_num_flag", attr.Bool(False))
            .attr("quant_mode", attr.Int(-1))
            .attr("active_expert_range", attr.ListInt([]))
            .attr("row_idx_type", attr.Int(0))
            .output(
                "expanded_x",
                "DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT8_E5M2, "
                "DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_INT4",
            )
            .output("expanded_row_idx", "DT_INT32")
            .output("expert_tokens_count_or_cumsum", "DT_INT64")
            .output("expanded_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("expanded_topk_weight", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_init_routing.default
    )
    def convert_moe_init_routing(
        x: Tensor,
        expert_idx: Tensor,
        *,
        scale: Tensor = None,
        offset: Tensor = None,
        topk_weight: Tensor = None,
        active_num: Union[int, Tensor] = -1,
        expert_capacity: int = -1,
        expert_num: int = -1,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 0,
        expert_tokens_num_flag: bool = False,
        quant_mode: int = -1,
        active_expert_range: list = None,
        row_idx_type: int = 0,
        x_dtype: int = None,
        meta_outputs: TensorSpec = None,
    ):
        active_num_tensor = dtype_promote(active_num, target_dtype=DataType.DT_INT64)

        return MoeInitRoutingV4(
            x=x,
            expert_idx=expert_idx,
            scale=scale,
            offset=offset,
            active_num=active_num_tensor,
            topk_weight=topk_weight,
            expert_capacity=expert_capacity,
            expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            quant_mode=quant_mode,
            active_expert_range=active_expert_range,
            row_idx_type=row_idx_type,
        )
