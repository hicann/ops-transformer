# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Torch registration and checked eager API for the DSL implementation."""

import torch
from cann_ops_transformer.op_builder import get_as_library
from .op_dsl.arch35 import mega_recurrent_kda as _implementation

InputValidation = _implementation.InputValidation


def _implementation_for_device(hidden_states):
    if hidden_states.device.type not in ("npu", "privateuseone"):
        return _implementation

    import cannbotdsl

    stream = torch.npu.current_stream(hidden_states.device)
    arch = cannbotdsl.get_platform_info(stream=stream).npu_arch
    if arch == "dav-9201":
        from .op_dsl.arch92 import mega_recurrent_kda as arch92

        return arch92
    return _implementation


_GRAPH_LIBRARY = get_as_library()
_GRAPH_LIBRARY.define(
    "mega_recurrent_kda(Tensor hidden_states, Tensor qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, Tensor conv_num_accepted_tokens, "
    "Tensor A_log, Tensor dt_bias, float scale, float lower_bound, float rms_norm_eps=1e-5, "
    "Tensor? cu_seqlens=None) -> (Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))"
)


@torch.library.impl(_GRAPH_LIBRARY, "mega_recurrent_kda", "Meta")
def _mega_recurrent_kda_meta(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    checkpoint_state,
    replay_u,
    replay_k,
    replay_decay,
    state_indices,
    conv_num_accepted_tokens,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    rms_norm_eps=1e-5,
    cu_seqlens=None,
):
    InputValidation.mega_metadata(locals())
    return (
        torch.empty_like(hidden_states, device="meta"),
        conv_state,
        replay_u,
        replay_k,
        replay_decay,
    )


@torch.library.impl(_GRAPH_LIBRARY, "mega_recurrent_kda", "PrivateUse1")
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-05,
    cu_seqlens: torch.Tensor | None = None,
):
    return _implementation_for_device(hidden_states).mega_recurrent_kda(
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        checkpoint_state,
        replay_u,
        replay_k,
        replay_decay,
        state_indices,
        conv_num_accepted_tokens,
        A_log,
        dt_bias,
        scale,
        lower_bound,
        rms_norm_eps,
        cu_seqlens,
    )
