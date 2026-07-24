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

"""TopK compare helpers for QuantLightningIndexer TestSpec adapters."""

import numpy as np

# Match the pytest boundary-score policy for non-identical TopK index sets.
_TOPK_REL_TOL = 0.0001
_VALUE_RTOL = 0.005
_VALUE_ATOL = 0.000025
_BFLOAT16_VALUE_ATOL = 0.0001
_VALUE_FAIL_RATIO = 0.005


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if "bfloat16" in str(value.dtype):
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def is_bfloat16(value):
    return "bfloat16" in str(getattr(value, "dtype", ""))


def row_set_diff_positions(output_row, golden_row):
    golden_set = {int(x) for x in golden_row if x >= 0}
    bad_positions = [idx for idx, value in enumerate(output_row) if value >= 0 and int(value) not in golden_set]
    output_set = {int(x) for x in output_row if x >= 0}
    missing_count = max(len(output_set - golden_set), len(golden_set - output_set))
    if missing_count <= 0:
        return []
    if len(bad_positions) >= missing_count:
        return bad_positions[:missing_count]
    positional = [idx for idx, (out, gold) in enumerate(zip(output_row, golden_row)) if out != gold]
    return (bad_positions + positional)[:missing_count]


def score_relative_diff(score, boundary):
    score_diff = abs(float(score) - float(boundary))
    denom = max(abs(float(boundary)), 1e-10)
    return abs(score_diff / denom)


def is_score_close_to_boundary(score, boundary):
    return score_relative_diff(score, boundary) <= _TOPK_REL_TOL


def row_score_diff_to_boundary(output_indices, golden_indices, score_row):
    if len(golden_indices) == 0:
        return 0.0, 0.0
    boundary_idx = int(golden_indices[-1])
    if boundary_idx < 0 or boundary_idx >= score_row.shape[-1]:
        return 0.0, 0.0
    boundary = float(score_row[boundary_idx])
    candidates = {int(x) for x in output_indices} ^ {int(x) for x in golden_indices}
    max_abs = 0.0
    max_rel = 0.0
    for idx in candidates:
        if 0 <= idx < score_row.shape[-1]:
            abs_diff = abs(float(score_row[idx]) - boundary)
            max_abs = max(max_abs, abs_diff)
            max_rel = max(max_rel, score_relative_diff(score_row[idx], boundary))
    return max_abs, max_rel


def is_row_equivalent_by_score(output_indices, golden_indices, score_row):
    if len(output_indices) != len(golden_indices) or len(golden_indices) == 0:
        return False
    output_set = {int(x) for x in output_indices}
    golden_set = {int(x) for x in golden_indices}
    if output_set == golden_set:
        return True
    only_output = list(output_set - golden_set)
    only_golden = list(golden_set - output_set)
    if len(only_output) != len(only_golden):
        return False

    boundary_idx = int(golden_indices[-1])
    if boundary_idx < 0 or boundary_idx >= score_row.shape[-1]:
        return False
    boundary = float(score_row[boundary_idx])
    for idx in only_output + only_golden:
        if idx < 0 or idx >= score_row.shape[-1]:
            return False
        if not is_score_close_to_boundary(score_row[idx], boundary):
            return False
    return True


def build_score_row_accessor(scores_np, index_shape, score_layout=None,
                             cu_seqlens_q=None):
    """Map output rows to the pytest score tensor without copying it."""
    prefix = tuple(index_shape[:-1])
    row_count = int(np.prod(prefix)) if prefix else 1
    if scores_np is None:
        return None, "score context missing"
    if tuple(scores_np.shape[:-1]) == prefix:
        score_rows = scores_np.reshape(row_count, scores_np.shape[-1])
        return lambda row_idx: score_rows[row_idx], None

    if score_layout == "BSND" and len(prefix) == 3 and scores_np.ndim == 4:
        batch_size, q_seq, head_num = prefix
        expected = (batch_size, head_num, q_seq)
        if tuple(scores_np.shape[:-1]) == expected:
            def get_bsnd_row(row_idx):
                batch_idx, remainder = divmod(row_idx, q_seq * head_num)
                q_idx, head_idx = divmod(remainder, head_num)
                return scores_np[batch_idx, head_idx, q_idx, :]

            return get_bsnd_row, None

    if score_layout == "TND" and len(prefix) == 2 and scores_np.ndim == 4:
        total_q, head_num = prefix
        if cu_seqlens_q is None:
            return None, "TND score context missing cu_seqlens_q"
        cu_q = as_numpy(cu_seqlens_q).reshape(-1).astype(np.int64)
        valid_prefix = (
            cu_q.size >= 2
            and int(cu_q[0]) == 0
            and int(cu_q[-1]) == total_q
            and np.all(cu_q[1:] >= cu_q[:-1])
        )
        max_q_seq = int(np.diff(cu_q).max()) if valid_prefix else -1
        valid_scores = (
            valid_prefix
            and scores_np.shape[0] == cu_q.size - 1
            and scores_np.shape[1] == head_num
            and scores_np.shape[2] >= max_q_seq
        )
        if valid_scores:
            def get_tnd_row(row_idx):
                q_idx, head_idx = divmod(row_idx, head_num)
                batch_idx = int(np.searchsorted(cu_q[1:], q_idx, side="right"))
                local_q_idx = q_idx - int(cu_q[batch_idx])
                return scores_np[batch_idx, head_idx, local_q_idx, :]

            return get_tnd_row, None

    reason = (
        f"score shape {scores_np.shape} vs index shape {index_shape}, "
        f"layout={score_layout}"
    )
    return None, reason


def topk_index_compare(npu_out, golden_out, scores, index_offsets=None,
                       score_layout=None, cu_seqlens_q=None):
    npu = as_numpy(npu_out)
    golden = as_numpy(golden_out)
    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"index shape mismatch: npu={npu.shape}, golden={golden.shape}",
        }
    if npu.size == 0:
        return {"pass": True, "precision": 100.0, "metrics": {"rows": 0}}

    scores_np = None if scores is None else as_numpy(scores)
    score_row_at, score_error = build_score_row_accessor(
        scores_np,
        golden.shape,
        score_layout=score_layout,
        cu_seqlens_q=cu_seqlens_q,
    )
    if score_row_at is None:
        sorted_npu = np.sort(npu, axis=-1).reshape(-1)
        sorted_golden = np.sort(golden, axis=-1).reshape(-1)
        diff_idx = np.where(sorted_npu != sorted_golden)[0].tolist()
        precision = (sorted_golden.size - len(diff_idx)) / sorted_golden.size * 100
        return {
            "pass": len(diff_idx) == 0,
            "precision": precision,
            "diff_indices": diff_idx[:1000],
            "error_info": None if not diff_idx else f"TopK fallback sorted-index compare failed: {score_error}",
            "metrics": {"fallback": score_error, "diff_positions": len(diff_idx)},
        }

    row_count = int(np.prod(golden.shape[:-1])) if golden.shape[:-1] else 1
    k_count = golden.shape[-1]
    output_rows = npu.reshape(row_count, k_count)
    golden_rows = golden.reshape(row_count, k_count)
    offset_rows = None
    if index_offsets is not None:
        offsets_np = as_numpy(index_offsets)
        if tuple(offsets_np.shape[:len(golden.shape[:-1])]) == tuple(golden.shape[:-1]):
            offset_rows = offsets_np.reshape(row_count, -1)[:, 0].astype(np.int64)

    diff_indices = []
    set_equal_rows = 0
    score_equiv_rows = 0
    failed_rows = 0
    max_failed_abs_diff = 0.0
    max_failed_rel_diff = 0.0
    for row_idx, (output_row, golden_row) in enumerate(zip(output_rows, golden_rows)):
        output_valid = output_row[output_row >= 0].astype(np.int64)
        golden_valid = golden_row[golden_row >= 0].astype(np.int64)
        if np.array_equal(np.sort(output_valid), np.sort(golden_valid)):
            set_equal_rows += 1
            continue
        score_output = output_valid
        score_golden = golden_valid
        if offset_rows is not None:
            offset = offset_rows[row_idx]
            score_output = output_valid - offset
            score_golden = golden_valid - offset
        score_row = score_row_at(row_idx)
        if is_row_equivalent_by_score(score_output, score_golden, score_row):
            score_equiv_rows += 1
            continue
        failed_rows += 1
        abs_diff, rel_diff = row_score_diff_to_boundary(score_output, score_golden, score_row)
        max_failed_abs_diff = max(max_failed_abs_diff, abs_diff)
        max_failed_rel_diff = max(max_failed_rel_diff, rel_diff)
        base = row_idx * k_count
        diff_indices.extend(base + i for i in row_set_diff_positions(output_row, golden_row))

    precision = (golden.size - len(diff_indices)) / golden.size * 100
    fail_ratio = len(diff_indices) / golden.size
    passed = failed_rows == 0
    error_info = None
    if failed_rows:
        verdict = "tolerated" if passed else "failed"
        error_info = (
            f"TopK index compare {verdict} rows={failed_rows}, diff_positions={len(diff_indices)}, "
            f"fail_ratio={fail_ratio:.6g}, max_failed_abs_score_diff={max_failed_abs_diff:.6g}, "
            f"max_failed_rel_score_diff={max_failed_rel_diff:.6g}"
        )
    return {
        "pass": passed,
        "precision": precision,
        "diff_indices": diff_indices[:1000],
        "error_info": error_info,
        "metrics": {
            "rows": row_count,
            "set_equal_rows": set_equal_rows,
            "score_equivalent_rows": score_equiv_rows,
            "failed_rows": failed_rows,
            "diff_positions": len(diff_indices),
            "fail_ratio": fail_ratio,
            "topk_rel_tol": _TOPK_REL_TOL,
            "max_failed_abs_score_diff": max_failed_abs_diff,
            "max_failed_rel_score_diff": max_failed_rel_diff,
        },
    }


def value_compare(npu_out, golden_out, scores=None):
    """Compare the returned sparse values against the pytest golden values."""
    del scores
    use_bfloat16_tolerance = is_bfloat16(npu_out) or is_bfloat16(golden_out)
    npu_raw = as_numpy(npu_out)
    golden_raw = as_numpy(golden_out)
    if npu_raw.shape != golden_raw.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"value shape mismatch: npu={npu_raw.shape}, golden={golden_raw.shape}",
        }
    if golden_raw.size == 0:
        return {"pass": True, "precision": 100.0}

    atol = _BFLOAT16_VALUE_ATOL if use_bfloat16_tolerance else _VALUE_ATOL
    row_width = golden_raw.shape[-1] if golden_raw.ndim else 1
    npu_rows = npu_raw.astype(np.float32).reshape(-1, row_width)
    golden_rows = golden_raw.astype(np.float32).reshape(-1, row_width)
    diff_idx = []
    failed_rows = 0
    for row_idx, (npu_row, golden_row) in enumerate(zip(npu_rows, golden_rows)):
        # Pytest sorts the NPU values before comparing each complete TopK row.
        npu_valid = np.sort(npu_row)[::-1]
        golden_valid = golden_row
        mismatch = ~np.isclose(
            npu_valid, golden_valid, rtol=_VALUE_RTOL, atol=atol, equal_nan=True)
        mismatch_indices = np.where(mismatch)[0]
        diff_idx.extend(row_idx * row_width + int(index) for index in mismatch_indices)
        if mismatch_indices.size / row_width > _VALUE_FAIL_RATIO:
            failed_rows += 1

    fail_ratio = len(diff_idx) / golden_raw.size
    precision = (golden_raw.size - len(diff_idx)) / golden_raw.size * 100
    return {
        "pass": failed_rows == 0,
        "precision": precision,
        "diff_indices": diff_idx[:1000],
        "error_info": None if not diff_idx else (
            f"value compare mismatches={len(diff_idx)}, fail_ratio={fail_ratio:.6g}"
        ),
        "metrics": {
            "rtol": _VALUE_RTOL,
            "atol": atol,
            "fail_ratio": fail_ratio,
            "row_fail_ratio_limit": _VALUE_FAIL_RATIO,
            "failed_rows": failed_rows,
        },
    }


def compare(*outputs, scores=None, index_offsets=None, score_layout=None,
            cu_seqlens_q=None):
    """Compare NPU outputs against golden references using TopK index and value tolerance policies."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {"pass": False, "precision": "invalid", "error_info": "compare expects NPU outputs followed by golden outputs"}
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    results = [topk_index_compare(
        npu_outputs[0],
        golden_outputs[0],
        scores,
        index_offsets=index_offsets,
        score_layout=score_layout,
        cu_seqlens_q=cu_seqlens_q,
    )]
    for idx in range(1, half):
        results.append(value_compare(npu_outputs[idx], golden_outputs[idx], scores=scores))
    return results[0] if len(results) == 1 else results
