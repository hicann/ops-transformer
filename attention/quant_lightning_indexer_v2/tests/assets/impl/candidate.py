# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Candidate-overload TTK assets using the existing CPU model and comparator."""

from types import SimpleNamespace

import numpy as np
import torch

from qli_v2_ttk_compare import COMPARATOR
from qli_v2_ttk_golden import CASE_DATA
from qli_v2_ttk_inputs import generate_qli_v2_inputs, rebuild_qli_v2_compare_data


def customize_inputs(*args, **kwargs):
    seed = int(np.random.randint(0, 2**32))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generate_qli_v2_inputs(*args, **kwargs)
    return None


def build_candidates(data, capacity, block_size):
    if capacity == -1 and block_size == -1:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)
    scores = data["scores"]
    block_count = (scores.shape[-1] + block_size - 1) // block_size
    padded = torch.full((*scores.shape[:-1], block_count * block_size), -torch.inf)
    padded[..., : scores.shape[-1]] = scores
    block_scores = padded.reshape(*scores.shape[:-1], block_count, block_size).amax(-1)
    if data["score_layout"] == "BSND":
        prefix = [batch * scores.shape[2] for batch in range(scores.shape[0] + 1)]
    else:
        prefix = data["cu_seqlens_q"].tolist()
    indices = torch.full((prefix[-1], 1, capacity), -1, dtype=torch.int32)
    lengths = torch.zeros((prefix[-1], 1), dtype=torch.int32)
    for batch, (start, end) in enumerate(zip(prefix, prefix[1:])):
        for local, row in enumerate(range(start, end)):
            token_scores = scores[batch, 0, local]
            valid = torch.isfinite(token_scores)
            visible = int(valid.sum())
            row_scores = block_scores[batch, 0, local]
            count = min(capacity, int(torch.isfinite(row_scores).sum()))
            # The partial logical block must survive Candidate truncation.
            if visible % block_size:
                row_scores[visible // block_size] = torch.inf
            if count:
                selected = row_scores.topk(count).indices
                indices[row, 0, :count] = selected.to(torch.int32)
            lengths[row, 0] = count
    if data["score_layout"] == "BSND":
        indices = indices.reshape(scores.shape[0], scores.shape[2], 1, capacity)
        lengths = lengths.reshape(scores.shape[0], scores.shape[2], 1)
    data["candidate_scores"] = block_scores
    data["candidate_indices"] = indices
    data["candidate_lengths"] = lengths
    return indices, lengths


def golden(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    candidate_topk_blocks,
    candidate_block_size,
    testcase_name=None,
    **kwargs,
):
    tensors = (
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
    )
    attributes = dict(kwargs, topk=topk, quant_mode=quant_mode)
    context = SimpleNamespace(input_tensors=tensors, attributes=attributes)
    data = rebuild_qli_v2_compare_data(context)
    indices, lengths = build_candidates(
        data, candidate_topk_blocks, candidate_block_size
    )
    CASE_DATA.put(testcase_name, data)
    values = (
        data["cpu_topk_value"]
        if attributes.get("return_value", 0)
        else torch.empty(0, dtype=torch.bfloat16)
    )
    return data["cpu_result"], values, indices, lengths


def contract_result(passed, detail):
    return {
        "pass": bool(passed),
        "precision": 100.0 if passed else 0.0,
        "error_info": None if passed else detail,
    }


def compare(*outputs, compare_context=None, **_kwargs):
    if len(outputs) != 8 or compare_context is None:
        raise ValueError("Candidate comparison requires four NPU and four CPU outputs")
    data = CASE_DATA.get(compare_context.testcase_name)
    if data is None:
        data = rebuild_qli_v2_compare_data(compare_context)
        build_candidates(
            data,
            int(compare_context.attributes["candidate_topk_blocks"]),
            int(compare_context.attributes["candidate_block_size"]),
        )
    actual = [COMPARATOR.to_torch(value) for value in outputs[:4]]
    expected = [COMPARATOR.to_torch(value) for value in outputs[4:]]
    assert actual[0].dtype == actual[2].dtype == actual[3].dtype == torch.int32
    results = COMPARATOR.compare(*outputs[:2], *outputs[4:6], compare_data=data)
    if not data["params"][-2]:
        results.append(
            contract_result(actual[1].numel() == 0, "disabled values must be empty")
        )
    expected_indices = expected[0].reshape_as(actual[0])
    valid = actual[0] != -1
    for row, reference in zip(
        actual[0].reshape(-1, actual[0].shape[-1]),
        expected_indices.reshape(-1, actual[0].shape[-1]),
    ):
        count = int((reference != -1).sum())
        assert (row[:count] != -1).all() and (row[count:] == -1).all(), (
            "invalid token prefix"
        )
        assert row[:count].unique().numel() == count, "duplicate token indices"
    if data["params"][-2]:
        assert torch.isneginf(actual[1][~valid]).all(), "invalid values must be -inf"
    if expected[2].numel() == 0:
        results.extend(
            contract_result(
                value.numel() == 0, "disabled Candidate output must be empty"
            )
            for value in actual[2:]
        )
        CASE_DATA.discard(data)
        return results
    capacity = expected[2].shape[-1]
    params = list(data["params"])
    params[2] = data["candidate_scores"].shape[-1]
    params[4] = params[2] * params[0]
    params[15] = [index * params[2] for index in range(params[0] + 1)]
    params[17] = [params[2]] * params[0]
    params[23], params[24], params[30], params[31] = capacity, 0, 1, 0
    block_data = dict(
        params=params, topk_value=data["candidate_scores"], output_idx_offset=None
    )
    results.extend(COMPARATOR.compare(outputs[2], outputs[6], compare_data=block_data))
    lengths = actual[3].reshape_as(expected[3])
    passed = torch.equal(lengths, expected[3])
    for row, count in zip(actual[2].reshape(-1, capacity), lengths.reshape(-1)):
        count = int(count)
        passed &= 0 <= count <= capacity
        if 0 <= count <= capacity:
            passed &= bool((row[:count] >= 0).all() and (row[count:] == -1).all())
            passed &= row[:count].unique().numel() == count
    results.append(
        contract_result(passed, "Candidate length/prefix/padding contract mismatch")
    )
    CASE_DATA.discard(data)
    return results
