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

import datetime
import math
import os
import sys
from time import time

import torch

import stem_indexer_golden


TOPK_BOUNDARY_RE_TOLERANCE = 1e-3
TOPK_BOUNDARY_ABS_TOLERANCE = 2.5e-5
MAX_BOUNDARY_RE_EXCEED_RATIO = 5e-3
MAX_PRINT_FAILURES = 8


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def cal_relative_diff_np_isclose(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + 1e-9)


COMPARE_SEPARATOR = "-" * 78
COMPARE_HEADER = (
    f"{'Loop':>10} {'ExpectOut':>16} {'RealOut':>16} {'FpDiff':>16} {'RateDiff':>16}"
)


def print_compare_row(loop, real_value, expect_value, rate_diff):
    expect_value = float(expect_value)
    real_value = float(real_value)
    abs_diff = abs(expect_value - real_value)
    print_log(
        f"{loop:>10d} {expect_value:>16.7f} {real_value:>16.7f} "
        f"{abs_diff:>16.7f} {rate_diff:>16.7f}"
    )


def display_output_np_isclose(real_data, expect_data, start, end):
    def display_inner(index):
        data_index = index + start
        rate_diff = cal_relative_diff_np_isclose(
            real_data[data_index], expect_data[data_index]
        )
        print_compare_row(
            start + index + 1,
            real_data[data_index],
            expect_data[data_index],
            rate_diff,
        )

    print_log(COMPARE_SEPARATOR)
    print_log(COMPARE_HEADER)
    print_log(COMPARE_SEPARATOR)
    split_count = int(end - start)
    if split_count <= 20:
        for index in range(split_count + 1):
            display_inner(index)
    else:
        for index in range(10):
            display_inner(index)
        print_log(f"{'...':>10} {'...':>16} {'...':>16} {'...':>16} {'...':>16}")
        for index in range(split_count - 10 + 1, split_count + 1):
            display_inner(index)


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    def print_error_row(index, rate_diff):
        print_compare_row(index, real_data[index], expect_data[index], rate_diff)

    print_log(f"Error Line{COMPARE_SEPARATOR[len('Error Line') :]}")
    print_log(COMPARE_HEADER)
    print_log(COMPARE_SEPARATOR)
    count = 0
    len_err = len(err_idx)
    for index in err_idx:
        count += 1
        if count < 10 or 90 < count < 100:
            print_error_row(index, relative_diff[count - 1])
        elif count == 10 or (count == 100 and len_err > 100):
            print_log(f"{'...':>10} {'...':>16} {'...':>16} {'...':>16} {'...':>16}")
        elif count > 100:
            break

    print_log(f"Max-RE line:{COMPARE_SEPARATOR[len('Max-RE line:') :]}")
    max_error = max(relative_diff)
    max_count = 0
    for position, rate_diff in enumerate(relative_diff):
        if rate_diff == max_error:
            print_error_row(err_idx[position], max_error)
            max_count += 1
            if max_count >= 3:
                break
    print_log(COMPARE_SEPARATOR)


def get_flattened_indices(actual_indices, expected_indices):
    return actual_indices.flatten().tolist(), expected_indices.flatten().tolist()


def display_index_output(actual_indices, expected_indices):
    real_data, expect_data = get_flattened_indices(actual_indices, expected_indices)
    if not real_data and not expect_data:
        return
    display_output_np_isclose(real_data, expect_data, 0, len(real_data) - 1)


def display_index_errors(actual_indices, expected_indices):
    real_data, expect_data = get_flattened_indices(actual_indices, expected_indices)
    err_idx = [
        index
        for index, (real_value, expect_value) in enumerate(zip(real_data, expect_data))
        if real_value != expect_value
    ]
    if not err_idx:
        return
    relative_diff = [
        cal_relative_diff_np_isclose(real_data[index], expect_data[index])
        for index in err_idx
    ]
    display_error_output(real_data, expect_data, err_idx, relative_diff)


def normalize_npu_result(npu_result):
    if isinstance(npu_result, (tuple, list)) and len(npu_result) == 2:
        return npu_result[0].detach().cpu().to(torch.int32), npu_result[
            1
        ].detach().cpu().to(torch.int32)
    raise TypeError("StemIndexer should return (sparse_indices, sparse_seq_len).")


def calculate_relative_error(value, reference):
    # Align with equal_nan=True.
    if math.isnan(value) and math.isnan(reference):
        return 0.0

    # Equal finite values and infinities with the same sign.
    if value == reference:
        return 0.0

    # One-sided NaN/Inf and infinities with different signs.
    if not math.isfinite(value) or not math.isfinite(reference):
        return float("inf")

    absolute_error = abs(value - reference)

    # Handle zero and small errors near zero.
    if absolute_error <= TOPK_BOUNDARY_ABS_TOLERANCE:
        return 0.0

    if reference == 0.0:
        return float("inf")

    return absolute_error / abs(reference)


def get_row_scores(case, inputs, b_idx, q_head_idx, q_block_idx):
    g_size = case["q_heads"] // case["kv_heads"]
    kv_head_idx = q_head_idx // g_size
    score_scale = 1.0 / ((case["stem_block_size"] // case["stem_stride"]) ** 2)
    q_vec = inputs["qflat"][b_idx, q_head_idx, q_block_idx].detach().cpu().float()
    k_group = inputs["kflat"][b_idx, kv_head_idx].detach().cpu().float()
    vbias = inputs["vbias"][b_idx, kv_head_idx].detach().cpu().float()
    return torch.matmul(k_group, q_vec) * score_scale + vbias


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

    raw_scores = get_row_scores(case, inputs, b_idx, q_head_idx, q_block_idx)
    topk_score_type = stem_indexer_golden.get_golden_topk_score_type(case)
    compare_scores = stem_indexer_golden.get_topk_sort_scores(
        raw_scores, topk_score_type
    )
    boundary_score = min(float(compare_scores[idx]) for idx in expected_dynamic)
    actual_differences = [
        {
            "idx": idx,
            "score": float(compare_scores[idx]),
            "boundary_score": boundary_score,
            "relative_error": calculate_relative_error(
                float(compare_scores[idx]), boundary_score
            ),
        }
        for idx in only_in_actual
    ]
    expected_differences = [
        {
            "idx": idx,
            "score": float(compare_scores[idx]),
            "boundary_score": boundary_score,
            "relative_error": calculate_relative_error(
                float(compare_scores[idx]), boundary_score
            ),
        }
        for idx in only_in_expected
    ]
    bad_actual_indices = [
        item
        for item in actual_differences
        if item["relative_error"] > TOPK_BOUNDARY_RE_TOLERANCE
    ]
    bad_expected_indices = [
        item
        for item in expected_differences
        if item["relative_error"] > TOPK_BOUNDARY_RE_TOLERANCE
    ]
    exceeded_difference_count = max(len(bad_actual_indices), len(bad_expected_indices))
    if exceeded_difference_count > 0:
        return {
            "b": b_idx,
            "q_head": q_head_idx,
            "q_block": q_block_idx,
            "actual": actual_prefix,
            "expected": expected_prefix,
            "bad_actual_indices": bad_actual_indices[:MAX_PRINT_FAILURES],
            "bad_expected_indices": bad_expected_indices[:MAX_PRINT_FAILURES],
            "relative_error_tolerance": TOPK_BOUNDARY_RE_TOLERANCE,
            "exceeded_difference_count": exceeded_difference_count,
            "reason": "dynamic TopK index difference is outside CPU boundary relative error tolerance",
        }
    return None


def assert_stem_indexer_result(
    expected_indices, expected_seq_len, npu_result, case, inputs=None
):
    start_time = time()
    actual_indices, actual_seq_len = normalize_npu_result(npu_result)
    expected_indices = expected_indices.to(torch.int32)
    expected_seq_len = expected_seq_len.to(torch.int32)

    print_log(f"total_line is {expected_seq_len.numel()}")
    if tuple(actual_seq_len.shape) != tuple(expected_seq_len.shape):
        print_log(
            f"{case['case_id']} sparse_seq_len shape mismatch: "
            f"actual={tuple(actual_seq_len.shape)}, expected={tuple(expected_seq_len.shape)}",
            level="ERROR",
        )
    assert tuple(actual_seq_len.shape) == tuple(expected_seq_len.shape), (
        f"{case['case_id']} sparse_seq_len shape mismatch: "
        f"actual={tuple(actual_seq_len.shape)}, expected={tuple(expected_seq_len.shape)}"
    )
    if tuple(actual_indices.shape) != tuple(expected_indices.shape):
        print_log(
            f"{case['case_id']} sparse_indices shape mismatch: "
            f"actual={tuple(actual_indices.shape)}, expected={tuple(expected_indices.shape)}",
            level="ERROR",
        )
    assert tuple(actual_indices.shape) == tuple(expected_indices.shape), (
        f"{case['case_id']} sparse_indices shape mismatch: "
        f"actual={tuple(actual_indices.shape)}, expected={tuple(expected_indices.shape)}"
    )
    if not torch.equal(actual_seq_len, expected_seq_len):
        seq_len_diff = torch.nonzero(actual_seq_len != expected_seq_len, as_tuple=False)
        print_log(
            f"{case['case_id']} sparse_seq_len mismatch: "
            f"diff_count={seq_len_diff.shape[0]}, "
            f"first={seq_len_diff[:MAX_PRINT_FAILURES].tolist()}",
            level="ERROR",
        )
    assert torch.equal(actual_seq_len, expected_seq_len), (
        f"{case['case_id']} sparse_seq_len mismatch"
    )

    failures = []
    exceeded_details = []
    mismatched_row_count = 0
    tolerated_row_count = 0
    exceeded_row_count = 0
    failed_row_count = 0
    total_valid_topk_count = 0
    total_exceeded_difference_count = 0
    padding_failures = []
    bad_padding_count = 0
    for index in torch.nonzero(expected_seq_len >= 0, as_tuple=False):
        b_idx, q_head_idx, q_block_idx = [int(item) for item in index]
        valid_len = int(expected_seq_len[b_idx, q_head_idx, q_block_idx])
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
        total_valid_topk_count += valid_len
        actual_prefix = actual_row[:valid_len].tolist()
        expected_prefix = expected_indices[
            b_idx, q_head_idx, q_block_idx, :valid_len
        ].tolist()
        if set(actual_prefix) != set(expected_prefix):
            mismatched_row_count += 1
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
                tolerated_row_count += 1
            elif "exceeded_difference_count" in failure:
                exceeded_row_count += 1
                total_exceeded_difference_count += failure["exceeded_difference_count"]
                if len(exceeded_details) < MAX_PRINT_FAILURES:
                    exceeded_details.append(failure)
            else:
                failed_row_count += 1
                if len(failures) < MAX_PRINT_FAILURES:
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

    exceeded_difference_ratio = (
        total_exceeded_difference_count / total_valid_topk_count
        if total_valid_topk_count > 0
        else 0.0
    )
    compare_pass = (
        bad_padding_count == 0
        and failed_row_count == 0
        and exceeded_difference_ratio <= MAX_BOUNDARY_RE_EXCEED_RATIO
    )
    display_index_output(actual_indices, expected_indices)
    print_log(f"耗时：{time() - start_time:.6f} 秒")
    if compare_pass:
        print_log("[success]StemIndexer TopK精度通过")
    else:
        print_log("[fail]StemIndexer TopK精度失败", level="ERROR")
    print_log(COMPARE_SEPARATOR)
    print_log(
        f"{'BadPadding':>12} {'FailedRows':>12} {'ExceedDiff':>12} "
        f"{'PctThd':>12} {'PctRlt':>12} {'Result':>10}"
    )
    print_log(COMPARE_SEPARATOR)
    print_log(
        f"{bad_padding_count:>12d} {failed_row_count:>12d} "
        f"{total_exceeded_difference_count:>12d} "
        f"{(1 - MAX_BOUNDARY_RE_EXCEED_RATIO) * 100:>11.6f}% "
        f"{(1 - exceeded_difference_ratio) * 100:>11.6f}% "
        f"{('Pass' if compare_pass else 'Failed'):>10}"
    )
    if not compare_pass:
        display_index_errors(actual_indices, expected_indices)

    assert bad_padding_count == 0, (
        f"{case['case_id']} sparse_indices padding mismatch: "
        f"bad_padding_count={bad_padding_count}, expected padding value=-1, failures={padding_failures}"
    )

    assert failed_row_count == 0, (
        f"{case['case_id']} sparse_indices valid prefix mismatch: "
        f"mismatched_row_count={mismatched_row_count}, "
        f"tolerated_row_count={tolerated_row_count}, failed_row_count={failed_row_count}, "
        f"failures={failures}"
    )

    assert exceeded_difference_ratio <= MAX_BOUNDARY_RE_EXCEED_RATIO, (
        f"{case['case_id']} sparse_indices TopK boundary mismatch ratio exceeds tolerance: "
        f"mismatched_row_count={mismatched_row_count}, "
        f"tolerated_row_count={tolerated_row_count}, exceeded_row_count={exceeded_row_count}, "
        f"total_exceeded_difference_count={total_exceeded_difference_count}, "
        f"total_valid_topk_count={total_valid_topk_count}, "
        f"exceeded_difference_ratio={exceeded_difference_ratio:.6g}, "
        f"relative_error_tolerance={TOPK_BOUNDARY_RE_TOLERANCE:.6g}, "
        f"max_exceeded_difference_ratio={MAX_BOUNDARY_RE_EXCEED_RATIO:.6g}, "
        f"details={exceeded_details}"
    )
