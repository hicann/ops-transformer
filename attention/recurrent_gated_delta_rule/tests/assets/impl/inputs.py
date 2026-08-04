#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK input adapter for RecurrentGatedDeltaRule (e2e + aclnn).

Fills index/length tensors (actual_seq_lengths, ssm_state_indices, num_accepted_tokens)
with shape-consistent values that cannot be randomly generated.
"""

__input__ = {
    "e2e": {"torch_npu.npu_recurrent_gated_delta_rule": "generate_rgdr_inputs"},
    "aclnn": {"aclnnRecurrentGatedDeltaRule": "aclnn_generate_rgdr_inputs"},
}

import torch


def fill_tensor(tensor, values):
    if tensor is None or values is None:
        return
    if torch.is_tensor(tensor):
        data = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(data.reshape(tensor.shape))
    else:
        import numpy as np

        tensor[:] = np.array(values, dtype=tensor.dtype).reshape(tensor.shape)


_INPUT_PARAM_ORDER = [
    "beta",
    "scale",
    "actual_seq_lengths",
    "ssm_state_indices",
    "num_accepted_tokens",
    "g",
    "gk",
]


def _extract_param(args, kwargs, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def generate_rgdr_inputs(query, key, value, state, *args, **kwargs):
    """Fill index/length tensors with values consistent with tensor shapes.

    - actual_seq_lengths: evenly split T (query.shape[0]) across B batches.
    - ssm_state_indices:  sequential arange(T).
    - num_accepted_tokens (optional): one valid value per batch in [1, seq_len].
    """
    actual_seq_lengths = _extract_param(args, kwargs, "actual_seq_lengths", 2)
    ssm_state_indices = _extract_param(args, kwargs, "ssm_state_indices", 3)
    num_accepted_tokens = _extract_param(args, kwargs, "num_accepted_tokens", 4)

    T = query.shape[0]
    B = actual_seq_lengths.shape[0]

    base = T // B
    remainder = T % B
    act_vals = [base + (1 if i < remainder else 0) for i in range(B)]
    fill_tensor(actual_seq_lengths, act_vals)

    fill_tensor(ssm_state_indices, list(range(T)))

    if num_accepted_tokens is not None:
        nat_vals = [min(i + 1, act_vals[i]) for i in range(B)]
        fill_tensor(num_accepted_tokens, nat_vals)


def aclnn_generate_rgdr_inputs(
    query,
    key,
    value,
    beta,
    stateRef,
    actualSeqLengths,
    ssmStateIndices,
    g=None,
    gk=None,
    numAcceptedTokens=None,
    scaleValue=0.125,
    out=None,
    **kwargs,
):
    """Fill index/length tensors for aclnn mode.

    Parameters follow aclnnRecurrentGatedDeltaRuleGetWorkspaceSize (without
    workspaceSize and executor). Modifies int32 tensors in-place; return value
    is ignored by the ACLNN input pipeline.
    """
    T = query.shape[0]
    B = actualSeqLengths.shape[0]

    base = T // B
    remainder = T % B
    act_vals = [base + (1 if i < remainder else 0) for i in range(B)]
    fill_tensor(actualSeqLengths, act_vals)

    fill_tensor(ssmStateIndices, list(range(T)))

    if numAcceptedTokens is not None:
        nat_vals = [min(i + 1, act_vals[i]) for i in range(B)]
        fill_tensor(numAcceptedTokens, nat_vals)
