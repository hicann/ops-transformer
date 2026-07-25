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

from collections import Counter

import torch

import stem_indexer_golden


TOPK_SCORE_ATOL = 2.5e-5
TOPK_SCORE_RTOL = 5e-3
MAX_INDEX_ERROR_RATIO = 5e-3
MAX_PRINT_FAILURES = 8


def normalize_npu_result(npu_result):
    if isinstance(npu_result, (tuple, list)) and len(npu_result) == 2:
        return npu_result[0].detach().cpu().to(torch.int32), npu_result[
            1
        ].detach().cpu().to(torch.int32)
    raise TypeError("StemIndexer should return (sparse_indices, sparse_seq_len).")


def count_index_mismatches(actual_indices, expected_indices):
    actual_counter = Counter(actual_indices)
    expected_counter = Counter(expected_indices)
    matched_count = sum((actual_counter & expected_counter).values())
    return max(len(actual_indices), len(expected_indices)) - matched_count


def get_row_scores(case, inputs, b_idx, q_head_idx, q_block_idx):
    qflat = inputs["qflat"].detach().cpu().float()
    kflat = inputs["kflat"].detach().cpu().float()
    vbias = inputs["vbias"].detach().cpu().float()
    g_size = case["q_heads"] // case["kv_heads"]
    kv_head_idx = q_head_idx // g_size
    score_scale = 1.0 / ((case["stem_block_size"] // case["stem_stride"]) ** 2)
    q_vec = qflat[b_idx, q_head_idx, q_block_idx]
    k_group = kflat[b_idx, kv_head_idx]
    return torch.matmul(k_group, q_vec) * score_scale + vbias[b_idx, kv_head_idx]


def get_s2_valid(case, b_idx, q_block_idx):
    q_len = case["q_seq_lens"][b_idx]
    kv_len = case["kv_seq_lens"][b_idx]
    prompt_len = case["num_prompt_tokens"][b_idx]
    q_block_num = stem_indexer_golden.ceil_div(q_len, case["stem_block_size"])
    kv_block_num = stem_indexer_golden.ceil_div(kv_len, case["stem_block_size"])
    decode = stem_indexer_golden.is_decode_case(q_len, kv_len, prompt_len)
    if case["causal"] and not decode:
        return stem_indexer_golden.calc_causal_s2_valid(
            q_block_idx, q_block_num, kv_block_num
        )
    return kv_block_num


def explain_topk_mismatch(
    case, inputs, b_idx, q_head_idx, q_block_idx, actual_prefix, expected_prefix
):
    if inputs is None:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "reason": "score inputs are unavailable",
        }

    s2_valid = get_s2_valid(case, b_idx, q_block_idx)
    actual_set = set(actual_prefix)
    expected_set = set(expected_prefix)

    if len(actual_set) != len(actual_prefix):
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "reason": "actual sparse_indices contains duplicate indices",
        }
    invalid_indices = [idx for idx in actual_prefix if idx < 0 or idx >= s2_valid]
    if invalid_indices:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "s2_valid": s2_valid,
            "invalid": invalid_indices[:MAX_PRINT_FAILURES],
            "reason": "actual sparse_indices contains out-of-range indices",
        }

    forced = stem_indexer_golden.get_forced_indices(
        s2_valid, case["initial_blocks"], case["window_size"]
    )
    missing_forced = sorted(forced - actual_set)
    if missing_forced:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "missing_forced": missing_forced,
            "reason": "actual sparse_indices misses forced indices",
        }

    expected_dynamic = sorted(expected_set - forced)
    actual_dynamic = sorted(actual_set - forced)
    only_in_actual = sorted(set(actual_dynamic) - set(expected_dynamic))
    only_in_expected = sorted(set(expected_dynamic) - set(actual_dynamic))
    if not expected_dynamic and only_in_actual:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "reason": "actual has dynamic indices while expected has none",
        }
    if not only_in_actual:
        return None

    scores = get_row_scores(case, inputs, b_idx, q_head_idx, q_block_idx)
    topk_score_type = stem_indexer_golden.get_golden_topk_score_type(case)
    sort_scores = stem_indexer_golden.get_topk_sort_scores(scores, topk_score_type)
    boundary_score = min(float(sort_scores[idx]) for idx in expected_dynamic)
    tolerance = max(abs(boundary_score) * TOPK_SCORE_RTOL, TOPK_SCORE_ATOL)
    bad_actual_indices = [
        {
            "idx": idx,
            "score": float(scores[idx]),
            "sort_score": float(sort_scores[idx]),
            "boundary_score": boundary_score,
            "golden_topk_score_type": topk_score_type,
            "tolerance": tolerance,
        }
        for idx in only_in_actual
        if float(sort_scores[idx]) + tolerance < boundary_score
    ]
    bad_expected_indices = [
        {
            "idx": idx,
            "score": float(scores[idx]),
            "sort_score": float(sort_scores[idx]),
            "boundary_score": boundary_score,
            "golden_topk_score_type": topk_score_type,
            "tolerance": tolerance,
        }
        for idx in only_in_expected
        if float(sort_scores[idx]) > boundary_score + tolerance
    ]
    if bad_actual_indices or bad_expected_indices:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "bad_actual_indices": bad_actual_indices[:MAX_PRINT_FAILURES],
            "bad_expected_indices": bad_expected_indices[:MAX_PRINT_FAILURES],
            "reason": "dynamic TopK index difference is outside CPU boundary score tolerance",
        }
    return None


def assert_stem_indexer_result(
    expected_indices, expected_seq_len, npu_result, case, inputs=None
):
    actual_indices, actual_seq_len = normalize_npu_result(npu_result)
    expected_indices = expected_indices.to(torch.int32)
    expected_seq_len = expected_seq_len.to(torch.int32)

    assert tuple(actual_seq_len.shape) == tuple(expected_seq_len.shape), (
        f"{case['case_id']} sparse_seq_len shape mismatch: "
        f"actual={tuple(actual_seq_len.shape)}, expected={tuple(expected_seq_len.shape)}"
    )
    assert tuple(actual_indices.shape) == tuple(expected_indices.shape), (
        f"{case['case_id']} sparse_indices shape mismatch: "
        f"actual={tuple(actual_indices.shape)}, expected={tuple(expected_indices.shape)}"
    )
    assert torch.equal(actual_seq_len, expected_seq_len), (
        f"{case['case_id']} sparse_seq_len mismatch"
    )

    failures = []
    bad_index_count = 0
    valid_index_count = 0
    padding_failures = []
    bad_padding_count = 0
    for index in torch.nonzero(expected_seq_len >= 0, as_tuple=False):
        b_idx, q_head_idx, q_block_idx = [int(item) for item in index]
        valid_len = int(expected_seq_len[b_idx, q_head_idx, q_block_idx])
        valid_index_count += valid_len
        actual_row = actual_indices[b_idx, q_head_idx, q_block_idx]
        invalid_positions = torch.nonzero(
            actual_row[valid_len:] != -1, as_tuple=False
        ).flatten()
        bad_padding_count += int(invalid_positions.numel())
        if invalid_positions.numel() > 0 and len(padding_failures) < MAX_PRINT_FAILURES:
            positions = (invalid_positions[:MAX_PRINT_FAILURES] + valid_len).tolist()
            padding_failures.append(
                {
                    "b": b_idx,
                    "q_head": q_head_idx,
                    "q_block": q_block_idx,
                    "positions": positions,
                    "actual": actual_row[positions].tolist(),
                }
            )
        if valid_len == 0:
            continue
        actual_prefix = actual_row[:valid_len].tolist()
        expected_prefix = expected_indices[
            b_idx, q_head_idx, q_block_idx, :valid_len
        ].tolist()
        if sorted(actual_prefix) != sorted(expected_prefix):
            row_bad_index_count = count_index_mismatches(actual_prefix, expected_prefix)
            bad_index_count += row_bad_index_count
            if len(failures) < MAX_PRINT_FAILURES:
                failure = explain_topk_mismatch(
                    case,
                    inputs,
                    b_idx,
                    q_head_idx,
                    q_block_idx,
                    actual_prefix,
                    expected_prefix,
                )
                if failure is None:
                    failure = {
                        "b": b_idx,
                        "q_head": q_head_idx,
                        "q_block": q_block_idx,
                        "actual": actual_prefix,
                        "expected": expected_prefix,
                        "reason": "index mismatch is within CPU boundary score tolerance",
                    }
                failure["bad_index_count"] = row_bad_index_count
                failures.append(failure)

    for index in torch.nonzero(expected_seq_len < 0, as_tuple=False):
        b_idx, q_head_idx, q_block_idx = [int(item) for item in index]
        actual_row = actual_indices[b_idx, q_head_idx, q_block_idx]
        invalid_positions = torch.nonzero(actual_row != -1, as_tuple=False).flatten()
        bad_padding_count += int(invalid_positions.numel())
        if invalid_positions.numel() > 0 and len(padding_failures) < MAX_PRINT_FAILURES:
            positions = invalid_positions[:MAX_PRINT_FAILURES].tolist()
            padding_failures.append(
                {
                    "b": b_idx,
                    "q_head": q_head_idx,
                    "q_block": q_block_idx,
                    "positions": positions,
                    "actual": actual_row[positions].tolist(),
                }
            )

    assert bad_padding_count == 0, (
        f"{case['case_id']} sparse_indices padding mismatch: "
        f"bad_padding_count={bad_padding_count}, expected padding value=-1, failures={padding_failures}"
    )

    index_error_ratio = (
        bad_index_count / valid_index_count if valid_index_count else 0.0
    )
    assert index_error_ratio <= MAX_INDEX_ERROR_RATIO, (
        f"{case['case_id']} sparse_indices valid prefix mismatch: "
        f"bad_index_count={bad_index_count}, "
        f"valid_index_count={valid_index_count}, index_error_ratio={index_error_ratio:.6g}, "
        f"max_index_error_ratio={MAX_INDEX_ERROR_RATIO:.6g}, failures={failures}"
    )
