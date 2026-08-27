# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class FusedCausalConv1dOpBuilder(OpBuilder):
    def __init__(self):
        super(FusedCausalConv1dOpBuilder, self).__init__(
            "fused_causal_conv1d", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/fused_causal_conv1d.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "fused_causal_conv1d("
            "Tensor x, Tensor weight, "
            "Tensor(a!) conv_states, "
            "*, "
            "Tensor? query_start_loc=None, "
            "Tensor? cache_indices=None, "
            "Tensor? initial_state_mode=None, "
            "Tensor? bias=None, "
            "Tensor? num_accepted_tokens=None, "
            "Tensor? num_computed_tokens=None, "
            "Tensor? block_idx_first_scheduled_token=None, "
            "Tensor? block_idx_last_scheduled_token=None, "
            "Tensor? initial_state_idx=None, "
            'str? activation="None", '
            "int? pad_slot_id=-1, "
            "int? max_query_len=-1, "
            "int? residual_connection=1, "
            "int? block_size=128, "
            "int? conv_mode=1, "
            "int? max_draft_tokens=7"
            ") -> Tensor"
        )

    def register_meta(self):
        """Register Meta implementation for shape/dtype inference."""

        @impl(get_as_library(), self.name, "Meta")
        def fused_causal_conv1d_meta(
            x: torch.Tensor,
            weight: torch.Tensor,
            conv_states: torch.Tensor,
            *,
            query_start_loc: Optional[torch.Tensor] = None,
            cache_indices: Optional[torch.Tensor] = None,
            initial_state_mode: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor],
            num_accepted_tokens: Optional[torch.Tensor] = None,
            num_computed_tokens: Optional[torch.Tensor] = None,
            block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
            block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
            initial_state_idx: Optional[torch.Tensor] = None,
            activation: str = "None",
            pad_slot_id: int = -1,
            max_query_len: int = -1,
            residual_connection: int = 1,
            block_size: int = 128,
            conv_mode: int = 1,
            max_draft_tokens: int = 7,
        ) -> torch.Tensor:
            y = torch.empty_like(x)
            return y


_fused_causal_conv1d_op_builder = FusedCausalConv1dOpBuilder()
_fused_causal_conv1d_op_builder._ensure_initialized()


@impl(get_as_library(), _fused_causal_conv1d_op_builder.name, "PrivateUse1")
def _fused_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    initial_state_mode: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    activation: str = "None",
    pad_slot_id: int = -1,
    max_query_len: int = -1,
    residual_connection: int = 1,
    block_size: int = 128,
    conv_mode: int = 1,
    max_draft_tokens: int = 7,
) -> torch.Tensor:
    _op_module = _fused_causal_conv1d_op_builder.load()
    return _op_module.fused_causal_conv1d(
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
        activation,
        pad_slot_id,
        max_query_len,
        residual_connection,
        block_size,
        conv_mode,
        max_draft_tokens,
    )


def fused_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    initial_state_mode: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    activation: str = "None",
    pad_slot_id: int = -1,
    max_query_len: int = -1,
    residual_connection: int = 1,
    block_size: int = 128,
    conv_mode: int = 1,
    max_draft_tokens: int = 7,
) -> torch.Tensor:
    return _fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
        bias=bias,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        activation=activation,
        pad_slot_id=pad_slot_id,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
        max_draft_tokens=max_draft_tokens,
    )
