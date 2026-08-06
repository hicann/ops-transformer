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

"""TTK input adapter for ChunkGatedDeltaRule.

Fills actual_seq_lengths with shape-consistent values and applies numerical
constraints required by the operator (L2-normalized q/k, g in [-1, 0], beta
in (0, 1)).
"""

__input__ = {
    "e2e": {"torch_npu.npu_chunk_gated_delta_rule": "generate_cgdr_inputs"},
    "aclnn": {"aclnnChunkGatedDeltaRule": "aclnn_chunk_gated_delta_rule_input"},
}

import torch
import torch.nn.functional as F


_PARAM_ORDER = [
    "beta",
    "initial_state",
    "actual_seq_lengths",
    "scale",
    "g",
]


def _extract_param(args, kwargs, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def fill_tensor(tensor, values):
    if tensor is None or values is None:
        return
    if torch.is_tensor(tensor):
        data = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(data.reshape(tensor.shape))


def _normalize_qk(query, key):
    """L2-normalize q and k along the last dim (in-place)."""
    if query is not None and torch.is_tensor(query):
        query.copy_(F.normalize(query.to(torch.float32), p=2, dim=-1).to(query.dtype))
    if key is not None and torch.is_tensor(key):
        key.copy_(F.normalize(key.to(torch.float32), p=2, dim=-1).to(key.dtype))


def _clip_tensor(tensor, lo, hi):
    if tensor is None or not torch.is_tensor(tensor):
        return
    tensor.copy_(tensor.to(torch.float32).clamp_(lo, hi).to(tensor.dtype))


def _fill_actual_seq_lengths(actual_seq_lengths, query):
    if actual_seq_lengths is None or not torch.is_tensor(actual_seq_lengths):
        return
    T = query.shape[0]
    B = actual_seq_lengths.shape[0]
    base = T // B
    remainder = T % B
    vals = [base + (1 if i < remainder else 0) for i in range(B)]
    fill_tensor(actual_seq_lengths, vals)


def _prepare_inputs(query, key, value, beta, initial_state, actual_seq_lengths, g):
    """Shared input preparation for e2e and aclnn adapters."""
    _normalize_qk(query, key)
    _fill_actual_seq_lengths(actual_seq_lengths, query)
    _clip_tensor(g, -1.0, 0.0)
    _clip_tensor(beta, 1e-6, 1.0 - 1e-6)


def generate_cgdr_inputs(query, key, value, *args, **kwargs):
    """Customize inputs for torch_npu.npu_chunk_gated_delta_rule.

    - L2-normalize q and k (operator requires pre-normalized inputs).
    - Fill actual_seq_lengths by evenly splitting T across B batches.
    - Clip g to [-1, 0] and beta to (0, 1) per operator constraints.
    """
    p = {}
    for i, name in enumerate(_PARAM_ORDER):
        p[name] = _extract_param(args, kwargs, name, i)

    _prepare_inputs(
        query,
        key,
        value,
        p["beta"],
        p["initial_state"],
        p["actual_seq_lengths"],
        p["g"],
    )


def aclnn_chunk_gated_delta_rule_input(
    query,
    key,
    value,
    beta,
    initialState,
    actualSeqLengths,
    gOptional,
    scaleValue,
    out,
    finalState,
    **kwargs,
):
    """Customize inputs for aclnnChunkGatedDeltaRule.

    Parameter names follow aclnnChunkGatedDeltaRuleGetWorkspaceSize (without
    workspaceSize and executor). Only input tensors are modified; output tensors
    (out, finalState) are left untouched.
    """
    _prepare_inputs(
        query,
        key,
        value,
        beta,
        initialState,
        actualSeqLengths,
        gOptional,
    )
