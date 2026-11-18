#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent Torch-composition golden/reference for MoeTokenUnpermuteWithEp.

Semantics (public README/docs) for the unpermute direction without padded mode:

    out[i] = sum_j ( permuted_tokens[sorted_indices[i * num_topk + j] - range[0]]
                     * (probs[i, j] if probs is not None else 1)
                     if range[0] <= sorted_indices[i * num_topk + j] < range[1]
                     else 0 )

`range`/`rangeOptional` empty (None or length 0) means the default {0, 0},
therefore every slot contributes 0 and the output is all zeros.

The golden keeps the compute dtype supplied by the test framework (when the
framework promotes inputs, e.g. fp16/bf16 -> fp32 and fp32 -> fp64, that promoted
dtype is used), accumulates in torch (never NumPy), and returns the result in the
input dtype. Both the kernel (GEIR) and the ACLNN two-phase API are registered here.
"""

from __future__ import annotations

import torch

__spec__ = {
    "moe_token_unpermute_with_ep": "MoeTokenUnpermuteWithEpNoProbsSpec",
    "aclnnMoeTokenUnpermuteWithEp": "MoeTokenUnpermuteWithEpAclnnSpec",
}

_TOLERANCE_L0 = {
    "float16": {"standard": "stat_rel_err", "level": "L0"},
    "bfloat16": {"standard": "stat_rel_err", "level": "L0"},
    "float32": {"standard": "stat_rel_err", "level": "L0"},
}

_TOLERANCE_L1 = {
    "float16": {"standard": "stat_rel_err", "level": "L1"},
    "bfloat16": {"standard": "stat_rel_err", "level": "L1"},
    "float32": {"standard": "stat_rel_err", "level": "L1"},
}


def _to_torch(value):
    """numpy / torch / tf carrier -> torch tensor, preserving dtype."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    from ttk.utilities.dtypes import numpy_to_torch_tensor

    return numpy_to_torch_tensor(value)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        from ttk.utilities.dtypes import torch_to_numpy_tensor

        return torch_to_numpy_tensor(value.detach().cpu())
    return value


def _normalize_range(range_value):
    """Return (start, end); empty/None maps to the documented default {0, 0}."""
    if range_value is None or isinstance(range_value, (int, float)):
        return 0, 0
    if isinstance(range_value, torch.Tensor):
        range_value = range_value.detach().cpu().reshape(-1).tolist()
    elif hasattr(range_value, "numpy"):
        range_value = range_value.numpy().reshape(-1).tolist()
    values = [int(v) for v in range_value]
    if len(values) == 0:
        return 0, 0
    if len(values) != 2:
        raise ValueError(f"range must contain exactly 2 elements, got {len(values)}")
    return values[0], values[1]


def _resolve_num_topk(num_topk, probs, indices):
    if probs is not None:
        if probs.dim() != 2:
            raise ValueError(
                f"probs must be 2D [tokens, num_topk], got rank {probs.dim()}"
            )
        return int(probs.shape[1])
    topk = int(num_topk)
    if topk < 1:
        raise ValueError(f"num_topk must be >= 1, got {topk}")
    return topk


def _reference(permuted_tokens, sorted_indices, probs, num_topk, range_value):
    tokens = _to_torch(permuted_tokens)
    indices = _to_torch(sorted_indices)
    probs_t = _to_torch(probs)

    if tokens is None or indices is None:
        raise ValueError("permuted_tokens and sorted_indices are required")
    if tokens.dim() != 2:
        raise ValueError(f"permuted_tokens must be 2D, got rank {tokens.dim()}")
    if tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError("permuted_tokens must not be empty")
    if indices.dim() != 1:
        raise ValueError(f"sorted_indices must be 1D, got rank {indices.dim()}")
    if indices.numel() == 0:
        raise ValueError("sorted_indices must not be empty")

    indices = indices.reshape(-1).to(torch.int64)
    topk = _resolve_num_topk(num_topk, probs_t, indices)
    total_length = indices.numel()
    if total_length % topk != 0:
        raise ValueError(
            f"sorted_indices length {total_length} must be divisible by num_topk {topk}"
        )
    token_count = total_length // topk
    if token_count < 1:
        raise ValueError("number of output tokens must be >= 1")
    if probs_t is not None:
        if probs_t.shape[0] != token_count:
            raise ValueError(
                f"probs dim0 {probs_t.shape[0]} must equal tokens {token_count}"
            )

    start, end = _normalize_range(range_value)
    if start < 0 or start > end:
        raise ValueError(f"invalid range [{start}, {end})")
    rows = int(tokens.shape[0])
    if end - start > rows:
        raise ValueError(
            f"permuted_tokens rows {rows} must be >= range span {end - start}"
        )

    out_dtype = tokens.dtype
    # Accumulate the reference in float64 so that finite values are not turned
    # into inf by intermediate overflow (e.g. near-max bf16 additions), then
    # round once to the declared output dtype.
    compute_dtype = torch.float64
    values = tokens.to(compute_dtype)

    valid = (indices >= start) & (indices < end)
    positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
    output = torch.zeros(
        (token_count, int(tokens.shape[1])), dtype=compute_dtype, device=tokens.device
    )
    if positions.numel() > 0:
        local = indices.index_select(0, positions) - start
        gathered = values.index_select(0, local)
        if probs_t is not None:
            weights = probs_t.to(compute_dtype).reshape(-1).index_select(0, positions)
            gathered = gathered * weights.unsqueeze(1)
        destination = torch.div(positions, topk, rounding_mode="floor")
        output.index_add_(0, destination, gathered)
    return output.to(out_dtype)


def _resolve_call(
    permuted_tokens=None,
    sorted_indices=None,
    probs=None,
    num_topk=1,
    range_value=None,
    permutedTokens=None,
    sortedIndices=None,
    probsOptional=None,
    numTopk=None,
    rangeOptional=None,
    **kwargs,
):
    del kwargs
    if permutedTokens is not None:
        permuted_tokens = permutedTokens
    if sortedIndices is not None:
        sorted_indices = sortedIndices
    if probsOptional is not None:
        probs = probsOptional
    if numTopk is not None:
        num_topk = numTopk
    if rangeOptional is not None:
        range_value = rangeOptional
    return _reference(permuted_tokens, sorted_indices, probs, num_topk, range_value)


def moe_token_unpermute_with_ep_golden(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range=None,
    padded_mode=False,
    restore_shape=None,
    unpermuted_tokens=None,
    **kwargs,
):
    del padded_mode, restore_shape, unpermuted_tokens
    kwargs.pop("numTopk", None)
    if kwargs.get("rangeOptional") is not None:
        range = kwargs["rangeOptional"]
    return [
        _to_numpy(
            _resolve_call(
                permuted_tokens=permuted_tokens,
                sorted_indices=sorted_indices,
                probs=probs,
                num_topk=num_topk,
                range_value=range,
            )
        )
    ]


def aclnn_moe_token_unpermute_with_ep_golden(
    permutedTokens,
    sortedIndices,
    probsOptional=None,
    numTopk=1,
    rangeOptional=None,
    paddedMode=False,
    restoreShapeOptional=None,
    out=None,
    **kwargs,
):
    del paddedMode, restoreShapeOptional, out
    return [
        _to_numpy(
            _resolve_call(
                permutedTokens=permutedTokens,
                sortedIndices=sortedIndices,
                probsOptional=probsOptional,
                numTopk=numTopk,
                rangeOptional=rangeOptional,
                **kwargs,
            )
        )
    ]


class _TorchMoeTokenUnpermuteWithEp:
    """Third-party reference class shared by the kernel and ACLNN paths."""

    def __call__(
        self,
        permuted_tokens=None,
        sorted_indices=None,
        probs=None,
        num_topk=1,
        range_value=None,
        permutedTokens=None,
        sortedIndices=None,
        probsOptional=None,
        numTopk=None,
        rangeOptional=None,
        **kwargs,
    ):
        return [
            _resolve_call(
                permuted_tokens=permuted_tokens,
                sorted_indices=sorted_indices,
                probs=probs,
                num_topk=num_topk,
                range_value=range_value,
                permutedTokens=permutedTokens,
                sortedIndices=sortedIndices,
                probsOptional=probsOptional,
                numTopk=numTopk,
                rangeOptional=rangeOptional,
                **kwargs,
            )
        ]


class _TorchAclnnMoeTokenUnpermuteWithEp:
    def __call__(
        self,
        permutedTokens=None,
        sortedIndices=None,
        probsOptional=None,
        numTopk=1,
        rangeOptional=None,
        paddedMode=False,
        restoreShapeOptional=None,
        out=None,
        **kwargs,
    ):
        del paddedMode, restoreShapeOptional, out
        return [
            _resolve_call(
                permutedTokens=permutedTokens,
                sortedIndices=sortedIndices,
                probsOptional=probsOptional,
                numTopk=numTopk,
                rangeOptional=rangeOptional,
                **kwargs,
            )
        ]


class MoeTokenUnpermuteWithEpNoProbsSpec:
    golden = staticmethod(moe_token_unpermute_with_ep_golden)
    third_party = {"torch": _TorchMoeTokenUnpermuteWithEp}
    tolerance = _TOLERANCE_L0


class MoeTokenUnpermuteWithEpAclnnSpec:
    golden = staticmethod(aclnn_moe_token_unpermute_with_ep_golden)
    third_party = {"torch": _TorchAclnnMoeTokenUnpermuteWithEp}
    tolerance = _TOLERANCE_L1
