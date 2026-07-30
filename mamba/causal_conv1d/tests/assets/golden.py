#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""CausalConv1d golden function."""

__golden__ = {"kernel": {"causal_conv1d": "CausalConv1d"}}

__input__ = {"kernel": {"causal_conv1d": "CausalConv1d_input"}}

import numpy as np
import torch
import torch.nn.functional as F
import ml_dtypes


def CausalConv1d_input(*input_arrays, **kwargs):
    """Custom input generator for causal_conv1d.

    Generates proper query_start_loc values (monotonically increasing sequence)
    and unique cache_indices to avoid duplicate cache slot mappings.
    """
    input_arrays = list(input_arrays)

    # Find query_start_loc in inputs (5th input, index 4)
    # Input order: x, weight, conv_states, bias, query_start_loc, ...
    if len(input_arrays) > 4 and input_arrays[4] is not None:
        qsl = input_arrays[4]
        if qsl.ndim == 1 and qsl.shape[0] > 1:
            batch_size = qsl.shape[0] - 1
            x = input_arrays[0]
            if x.ndim == 2:
                total_tokens = x.shape[0]
            else:
                total_tokens = x.shape[0] * x.shape[1]

            testcase_name = kwargs.get("testcase_name", "")
            seq_len = None
            if "_s" in testcase_name:
                import re

                match = re.search(r"_s(\d+)_", testcase_name)
                if match:
                    seq_len = int(match.group(1))

            if seq_len is None:
                seq_len = total_tokens // batch_size

            qsl_values = np.arange(batch_size + 1, dtype=np.int32) * seq_len
            qsl_values[-1] = total_tokens
            input_arrays[4] = qsl_values

    # Generate unique cache_indices (6th input, index 5) to avoid duplicate cache slots
    if len(input_arrays) > 5 and input_arrays[5] is not None:
        ci = input_arrays[5]
        if ci.ndim == 1 and ci.shape[0] > 0:
            batch_size = ci.shape[0]
            ci_values = np.arange(0, batch_size, dtype=np.int32)
            input_arrays[5] = ci_values

    return input_arrays


def _to_torch(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    try:
        return torch.from_numpy(arr)
    except TypeError:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _to_numpy(t):
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _compute_conv1d(seq_input, history, weight_t, bias_t, activation, dtype, device):
    """Causal conv1d via F.conv1d with depthwise groups."""
    padded = torch.cat([history, seq_input], dim=0)
    weight_view = weight_t.T.unsqueeze(1)
    result = (
        F.conv1d(
            padded.T.unsqueeze(0),
            weight_view,
            bias=bias_t,
            stride=1,
            padding=0,
            groups=seq_input.shape[-1],
        )
        .squeeze(0)
        .T
    )
    if activation:
        result = _silu(result)
    return result.to(dtype)


def _causal_conv1d_fn(
    x,
    weight,
    conv_states,
    bias,
    query_start_loc,
    cache_indices,
    initial_state_mode,
    activation_mode,
    null_block_id,
    **extra,
):
    """Prefill mode golden — matches vllm causal_conv1d_fn semantics.

    conv_states write-back: write the last state_len tokens of [history || seq] to [0:state_len].
    When initial_state_mode[i]=0 (cold start), history is zero-filled.
    """
    x_t = _to_torch(x)
    orig_dtype = x_t.dtype
    x_t = x_t.to(torch.float32)
    weight_t = _to_torch(weight).to(torch.float32)
    bias_t = _to_torch(bias)
    if bias_t is not None:
        bias_t = bias_t.to(torch.float32)
    conv_states_t = _to_torch(conv_states)
    query_start_loc_t = _to_torch(query_start_loc)
    cache_indices_t = _to_torch(cache_indices)
    init_state_mode_t = _to_torch(initial_state_mode)

    activation = activation_mode == "silu"
    kernel_size = weight_t.shape[0]
    dtype = orig_dtype
    device = x_t.device

    if x_t.ndim == 3:
        batch_size = x_t.shape[0]
        seq_list = [x_t[i] for i in range(batch_size)]
        is_3d = True
    else:
        query_starts = query_start_loc_t.to(torch.int64)
        batch_size = query_starts.shape[0] - 1
        seq_list = [
            x_t[query_starts[i] : query_starts[i + 1]] for i in range(batch_size)
        ]
        is_3d = False

    cache_indices_list = (
        cache_indices_t.tolist()
        if cache_indices_t is not None
        else list(range(batch_size))
    )
    init_state_modes = (
        init_state_mode_t.tolist()
        if init_state_mode_t is not None
        else [0] * batch_size
    )
    state_len = conv_states_t.shape[1]
    conv_states_out = conv_states_t.clone()
    output_list = []

    for i in range(batch_size):
        cache_idx = int(cache_indices_list[i])
        if cache_indices_t is not None and cache_idx == null_block_id:
            cur_seq_len = x_t.shape[1] if is_3d else seq_list[i].shape[0]
            output_list.append(
                torch.zeros(
                    cur_seq_len, x_t.shape[-1], dtype=torch.float32, device=device
                )
            )
            continue

        seq_input = seq_list[i]
        has_init = bool(init_state_modes[i])
        if has_init:
            history = conv_states_t[cache_idx][0 : kernel_size - 1].to(torch.float32)
        else:
            history = torch.zeros(
                kernel_size - 1, seq_input.shape[-1], dtype=torch.float32, device=device
            )

        result = _compute_conv1d(
            seq_input, history, weight_t, bias_t, activation, dtype, device
        )
        output_list.append(result)

        write_history = (
            history
            if has_init
            else torch.zeros(
                kernel_size - 1, seq_input.shape[-1], dtype=torch.float32, device=device
            )
        )
        write_padded = torch.cat([write_history, seq_input], dim=0)
        conv_states_out[cache_idx][0:state_len] = write_padded[-state_len:].to(dtype)

    output = torch.stack(output_list, dim=0) if is_3d else torch.cat(output_list, dim=0)
    return _to_numpy(conv_states_out), _to_numpy(output)


def _causal_conv1d_update(
    x,
    weight,
    conv_states,
    bias,
    query_start_loc,
    cache_indices,
    num_accepted_tokens,
    activation_mode,
    null_block_id,
    **extra,
):
    """Update mode golden — matches vllm causal_conv1d_update conv_state sliding window.

    vllm conv_state update:
      effective_state_len = width-1 + seqlen-1  (spec decode) or width-1 (normal)
      new_state = [old_state[offset+1 : offset+effective_state_len], x[0:seqlen]]
      write new_state[0:effective_state_len] to conv_states[cache_idx][0:effective_state_len]

    For spec decode (num_accepted_tokens provided):
      offset = num_accepted_tokens[i] - 1  (sliding window shift)
      effective_state_len = width-1 + seqlen-1
    For normal update:
      offset = 0
      effective_state_len = width-1
    """
    x_t = _to_torch(x)
    orig_dtype = x_t.dtype
    x_t = x_t.to(torch.float32)
    weight_t = _to_torch(weight).to(torch.float32)
    bias_t = _to_torch(bias)
    if bias_t is not None:
        bias_t = bias_t.to(torch.float32)
    conv_states_t = _to_torch(conv_states)
    query_start_loc_t = _to_torch(query_start_loc)
    cache_indices_t = _to_torch(cache_indices)
    accepted_tokens_t = _to_torch(num_accepted_tokens)

    activation = activation_mode == "silu"
    kernel_size = weight_t.shape[0]
    dtype = orig_dtype
    device = x_t.device

    if x_t.ndim == 3:
        batch_size = x_t.shape[0]
        seq_list = [x_t[i] for i in range(batch_size)]
        is_3d = True
    else:
        query_starts = query_start_loc_t.to(torch.int64)
        batch_size = query_starts.shape[0] - 1
        seq_list = [
            x_t[query_starts[i] : query_starts[i + 1]] for i in range(batch_size)
        ]
        is_3d = False

    cache_indices_list = (
        cache_indices_t.tolist()
        if cache_indices_t is not None
        else list(range(batch_size))
    )
    state_len = conv_states_t.shape[1]
    conv_states_out = conv_states_t.clone()
    output_list = []

    for i in range(batch_size):
        cache_idx = int(cache_indices_list[i])
        if cache_indices_t is not None and cache_idx == null_block_id:
            cur_seq_len = x_t.shape[1] if is_3d else seq_list[i].shape[0]
            output_list.append(
                torch.zeros(
                    cur_seq_len, x_t.shape[-1], dtype=torch.float32, device=device
                )
            )
            continue

        seq_input = seq_list[i]
        seqlen_i = seq_input.shape[0]

        if accepted_tokens_t is not None:
            offset = max(int(accepted_tokens_t[i]) - 1, 0)
            effective_state_len = kernel_size - 1 + seqlen_i - 1
        else:
            offset = 0
            effective_state_len = kernel_size - 1

        history = conv_states_t[cache_idx][offset : offset + kernel_size - 1].to(
            torch.float32
        )

        result = _compute_conv1d(
            seq_input, history, weight_t, bias_t, activation, dtype, device
        )
        output_list.append(result)

        old_keep = kernel_size - 2
        new_state = torch.cat(
            [
                conv_states_t[cache_idx][offset + 1 : offset + 1 + old_keep].to(
                    torch.float32
                ),
                seq_input,
            ],
            dim=0,
        )
        write_len = min(effective_state_len, state_len)
        conv_states_out[cache_idx][0:write_len] = new_state[0:write_len].to(dtype)

    output = torch.stack(output_list, dim=0) if is_3d else torch.cat(output_list, dim=0)
    return _to_numpy(conv_states_out), _to_numpy(output)


def CausalConv1d(
    x,
    weight,
    conv_states,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    *,
    activation_mode="silu",
    null_block_id=0,
    **extra,
):
    """Golden function for CausalConv1d operator.

    Routes to fn (prefill) or update (decode) based on input shapes,
    matching vllm's InferIsFnMode logic.
    """
    x_t = _to_torch(x)
    is_fn_mode = False
    if num_accepted_tokens is not None:
        is_fn_mode = False
    elif x_t.ndim == 3:
        is_fn_mode = x_t.shape[1] > 1
    elif x_t.ndim == 2:
        is_fn_mode = query_start_loc is not None

    if is_fn_mode:
        return _causal_conv1d_fn(
            x,
            weight,
            conv_states,
            bias,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            activation_mode,
            null_block_id,
            **extra,
        )
    else:
        return _causal_conv1d_update(
            x,
            weight,
            conv_states,
            bias,
            query_start_loc,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
            null_block_id,
            **extra,
        )
