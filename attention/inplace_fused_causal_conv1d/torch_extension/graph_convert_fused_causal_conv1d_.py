# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
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


_ACTIVATION_MAP = {
    "silu": "silu",
    "none": "none",
}


if _TORCHAIR_AVAILABLE:

    def _parse_activation(activation):
        if activation == "silu":
            return 1
        elif activation == "swish":
            return 2
        return 0

    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [False, False, False, True, True, True, True, True, True, True, True, True],
    )
    def InplaceFusedCausalConv1d(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: Optional[torch.Tensor] = None,
        cache_indices: Optional[torch.Tensor] = None,
        initial_state_mode: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        num_computed_tokens: Optional[torch.Tensor] = None,
        block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
        block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
        initial_state_idx: Optional[torch.Tensor] = None,
        *,
        activation: str = "None",
        pad_slot_id: int = -1,
        max_query_len: int = -1,
        residual_connection: int = 1,
        block_size: int = 128,
        conv_mode: int = 1,
        max_draft_tokens: int = 7,
        null_block_id: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(InplaceFusedCausalConv1d)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
        .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
        .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
        .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
        .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(num_computed_tokens, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(block_idx_first_scheduled_token, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(block_idx_last_scheduled_token, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(initial_state_idx, TensorType({DT_INT32}))
        .ATTR(activation_mode, Int, 0)
        .ATTR(pad_slot_id, Int, -1)
        .ATTR(max_query_len, Int, -1)
        .ATTR(residual_connection, Int, 1)
        .ATTR(block_size, Int, 128)
        .ATTR(conv_mode, Int, 1)
        .ATTR(max_draft_tokens, Int, 7)
        .OUTPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
        .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
        .OP_END_FACTORY_REG(InplaceFusedCausalConv1d)
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "weight": weight,
            "conv_states": conv_states,
            "query_start_loc": query_start_loc,
            "cache_indices": cache_indices,
            "initial_state_mode": initial_state_mode,
            "bias": bias,
            "num_accepted_tokens": num_accepted_tokens,
            "num_computed_tokens": num_computed_tokens,
            "block_idx_first_scheduled_token": block_idx_first_scheduled_token,
            "block_idx_last_scheduled_token": block_idx_last_scheduled_token,
            "initial_state_idx": initial_state_idx,
        }
        attrs = {
            "activation": _parse_activation(activation),
            "pad_slot_id": pad_slot_id,
            "max_query_len": max_query_len,
            "residual_connection": residual_connection,
            "block_size": block_size,
            "conv_mode": conv_mode,
            "max_draft_tokens": max_draft_tokens,
            "null_block_id": attr.Int(null_block_id),
        }
        outputs = ["conv_states", "x"]

        return ge_op(
            op_type="InplaceFusedCausalConv1d",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("InplaceFusedCausalConv1d")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("weight", "DT_BF16, DT_FLOAT16")
            .input("conv_states", "DT_BF16, DT_FLOAT16")
            .optional_input("bias", "DT_BF16, DT_FLOAT16")
            .optional_input("query_start_loc", "DT_INT32")
            .optional_input("cache_indices", "DT_INT32")
            .optional_input("initial_state_mode", "DT_INT32")
            .optional_input("num_accepted_tokens", "DT_INT32")
            .optional_input("num_computed_tokens", "DT_INT32")
            .optional_input("block_idx_first_scheduled_token", "DT_INT32")
            .optional_input("block_idx_last_scheduled_token", "DT_INT32")
            .optional_input("initial_state_idx", "DT_INT32")
            .attr("activation_mode", _parse_activation(activation))
            .attr("pad_slot_id", attr.Int(-1))
            .attr("max_query_len", attr.Int(-1))
            .attr("residual_connection", attr.Int(1))
            .attr("block_size", attr.Int(128))
            .attr("conv_mode", conv_mode)
            .attr("max_draft_tokens", attr.Int(7))
            .attr("null_block_id", attr.Int(0))
            .output("conv_states", "DT_BF16, DT_FLOAT16")
            .output("x", "DT_BF16, DT_FLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.inplace_fused_causal_conv1d.default
    )
    def converter_inplace_fused_causal_conv1d(
        x: Tensor,
        weight: Tensor,
        conv_states: Tensor,
        *,
        query_start_loc: Optional[Tensor] = None,
        cache_indices: Optional[Tensor] = None,
        initial_state_mode: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        num_accepted_tokens: Optional[Tensor] = None,
        num_computed_tokens: Optional[Tensor] = None,
        block_idx_first_scheduled_token: Optional[Tensor] = None,
        block_idx_last_scheduled_token: Optional[Tensor] = None,
        initial_state_idx: Optional[Tensor] = None,
        activation: Optional[str] = "None",
        pad_slot_id: Optional[int] = -1,
        max_query_len: Optional[int] = -1,
        residual_connection: Optional[int] = 0,
        block_size: Optional[int] = 128,
        conv_mode: Optional[int] = 1,
        max_draft_tokens: Optional[int] = 7,
        null_block_id: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        _conv_states_out, x_out = InplaceFusedCausalConv1d(
            x,
            weight,
            conv_states,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            bias,
            num_accepted_tokens,
            num_computed_tokens,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            activation_mode=_parse_activation(activation),
            pad_slot_id=pad_slot_id,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
            max_draft_tokens=max_draft_tokens,
            null_block_id=null_block_id,
        )
        return x_out


else:

    def convert_inplace_fused_causal_conv1d(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
