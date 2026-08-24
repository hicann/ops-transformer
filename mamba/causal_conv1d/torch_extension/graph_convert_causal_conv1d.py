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

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True],
    )
    def CausalConv1d(
        x: Tensor,
        weight: Tensor,
        conv_states: Tensor,
        bias: Optional[Tensor] = None,
        query_start_loc: Optional[Tensor] = None,
        cache_indices: Optional[Tensor] = None,
        initial_state_mode: Optional[Tensor] = None,
        num_accepted_tokens: Optional[Tensor] = None,
        *,
        activation_mode: str = "silu",
        null_block_id: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(CausalConv1d)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
        .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
        .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
        .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
        .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
        .ATTR(activation_mode, String, "silu")
        .ATTR(null_block_id, Int, 0)
        .OUTPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "weight": weight,
            "conv_states": conv_states,
            "bias": bias,
            "query_start_loc": query_start_loc,
            "cache_indices": cache_indices,
            "initial_state_mode": initial_state_mode,
            "num_accepted_tokens": num_accepted_tokens,
        }
        attrs = {
            "activation_mode": attr.Str(activation_mode),
            "null_block_id": attr.Int(null_block_id),
        }
        outputs = ["conv_states", "y"]

        return ge_op(
            op_type="CausalConv1d",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("CausalConv1d")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("weight", "DT_BF16, DT_FLOAT16")
            .input("conv_states", "DT_BF16, DT_FLOAT16")
            .optional_input("bias", "DT_BF16, DT_FLOAT16")
            .optional_input("query_start_loc", "DT_INT32")
            .optional_input("cache_indices", "DT_INT32")
            .optional_input("initial_state_mode", "DT_INT32")
            .optional_input("num_accepted_tokens", "DT_INT32")
            .attr("activation_mode", attr.Str("silu"))
            .attr("null_block_id", attr.Int(0))
            .output("conv_states", "DT_BF16, DT_FLOAT16")
            .output("y", "DT_BF16, DT_FLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.causal_conv1d_fn.default
    )
    def convert_causal_conv1d_fn(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        conv_states: Tensor,
        query_start_loc: Tensor,
        *,
        cache_indices: Optional[Tensor] = None,
        has_initial_state: Optional[Tensor] = None,
        activation: str = "silu",
        pad_slot_id: int = -1,
        null_block_id: int = 0,
        block_idx_first_scheduled_token: Optional[Tensor] = None,
        block_idx_last_scheduled_token: Optional[Tensor] = None,
        initial_state_idx: Optional[Tensor] = None,
        num_computed_tokens: Optional[Tensor] = None,
        block_size_to_align: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        activation_mode = _ACTIVATION_MAP.get(activation, activation)
        conv_states_out, y = CausalConv1d(
            x=x,
            weight=weight,
            conv_states=conv_states,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=has_initial_state,
            num_accepted_tokens=None,
            activation_mode=activation_mode,
            null_block_id=null_block_id,
        )
        return y

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.causal_conv1d_update.default
    )
    def convert_causal_conv1d_update(
        x: Tensor,
        conv_state: Tensor,
        weight: Tensor,
        *,
        bias: Optional[Tensor] = None,
        activation: str = "silu",
        conv_state_indices: Optional[Tensor] = None,
        num_accepted_tokens: Optional[Tensor] = None,
        query_start_loc: Optional[Tensor] = None,
        max_query_len: int = -1,
        null_block_id: int = 0,
        block_idx_last_scheduled_token: Optional[Tensor] = None,
        initial_state_idx: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None,
    ):
        activation_mode = _ACTIVATION_MAP.get(activation, activation)
        conv_states_out, y = CausalConv1d(
            x=x,
            weight=weight,
            conv_states=conv_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=conv_state_indices,
            initial_state_mode=None,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=activation_mode,
            null_block_id=null_block_id,
        )
        return y

else:

    def convert_causal_conv1d_fn(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )

    def convert_causal_conv1d_update(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
