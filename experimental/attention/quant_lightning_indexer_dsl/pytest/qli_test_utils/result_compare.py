# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Optional

import torch


TOPK_BOUNDARY_RTOL = 0.001
VALUE_RTOL = 0.005
VALUE_ATOL = 2.5e-5
BF16_VALUE_ATOL = 1.0e-4
BF16_VALUE_RTOL = 1.0 / 128.0
VALUE_FAILURE_RATIO = 0.005
COMPARE_ROW_CHUNK = 256


def _values_close(
    real_data: torch.Tensor,
    expect_data: torch.Tensor,
    *,
    force_bf16: bool = False,
) -> bool:
    """Apply the QLI value tolerance and allowed failure ratio."""
    real = real_data.detach().cpu()
    expect = expect_data.detach().cpu()
    is_bfloat16 = force_bf16 or real.dtype == torch.bfloat16
    atol = BF16_VALUE_ATOL if is_bfloat16 else VALUE_ATOL
    rtol = BF16_VALUE_RTOL if is_bfloat16 else VALUE_RTOL
    close = torch.isclose(
        real.to(torch.float32),
        expect.to(torch.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    if close.numel() == 0:
        return True
    return float(close.sum().item()) / float(close.numel()) >= 1.0 - VALUE_FAILURE_RATIO


def _compare_topk_row(
    cpu_indices: torch.Tensor,
    npu_indices: torch.Tensor,
    topk_value: torch.Tensor,
    *,
    cpu_output_row: Optional[torch.Tensor] = None,
    npu_output_row: Optional[torch.Tensor] = None,
    return_value: bool = False,
    boundary_rtol: float = TOPK_BOUNDARY_RTOL,
) -> tuple[bool, float]:
    """Apply the QLI TopK boundary-equivalence rule to one output row."""
    cur_cpu = [int(x) for x in cpu_indices.tolist()]
    cur_npu = [int(x) for x in npu_indices.tolist()]
    score_size = int(topk_value.numel())
    if (
        len(cur_cpu) != len(cur_npu)
        or any(x < 0 or x >= score_size for x in cur_cpu)
        or any(x < 0 or x >= score_size for x in cur_npu)
        or len(set(cur_cpu)) != len(cur_cpu)
        or len(set(cur_npu)) != len(cur_npu)
    ):
        return False, float("inf")
    cpu_set = set(cur_cpu)
    npu_set = set(cur_npu)
    if cpu_set == npu_set:
        return True, 0.0
    if not cur_cpu or len(cpu_set - npu_set) != len(npu_set - cpu_set):
        return False, float("inf")

    value_bm = float(topk_value[cur_cpu[-1]])
    only_in_npu = list(npu_set - cpu_set)
    only_in_cpu = list(cpu_set - npu_set)
    max_re = 0.0
    for diff_idx in range(len(only_in_npu)):
        npu_idx = only_in_npu[diff_idx]
        cpu_idx = only_in_cpu[diff_idx]
        element_npu = float(topk_value[npu_idx])
        element_cpu = float(topk_value[cpu_idx])
        npu_ae = abs(element_npu - value_bm)
        cpu_ae = abs(element_cpu - value_bm)
        if value_bm == 0.0:
            npu_re = 0.0 if npu_ae == 0.0 else float("inf")
            cpu_re = 0.0 if cpu_ae == 0.0 else float("inf")
        else:
            npu_re = abs(npu_ae / value_bm)
            cpu_re = abs(cpu_ae / value_bm)
        if npu_re > boundary_rtol or cpu_re > boundary_rtol:
            # Returned values provide the second gate when membership differs
            # gate to the complete output row before declaring the row bad.
            if not return_value:
                return False, max(max_re, npu_re, cpu_re)
            if cpu_output_row is None or npu_output_row is None:
                return False, max(max_re, npu_re, cpu_re)
            # QLI V2 converts returned BF16 values to FP32 NumPy before this
            # after the TopK boundary check.
            if not _values_close(
                npu_output_row.to(torch.float32), cpu_output_row.to(torch.float32)
            ):
                return False, max(max_re, npu_re, cpu_re)
        max_re = max(max_re, npu_re, cpu_re)
    return True, max_re


def _reference_row(reference, row: int, *, candidate: bool) -> torch.Tensor:
    """Materialize one score row only when boundary fallback is required."""
    if isinstance(reference, torch.Tensor):
        flat = (
            reference.detach().cpu().to(torch.float32).reshape(-1, reference.shape[-1])
        )
        return flat[row]
    method = "candidate_scores" if candidate else "token_scores"
    scorer = getattr(reference, method, None)
    if scorer is None:
        raise TypeError(f"reference score provider is missing {method}()")
    return scorer(row).detach().cpu().to(torch.float32)


def _compare_sparse_indices(
    expected_indices: torch.Tensor,
    actual_indices: torch.Tensor,
    reference_scores,
    expected_values: Optional[torch.Tensor],
    actual_values: Optional[torch.Tensor],
    *,
    return_value: bool,
) -> dict[str, object]:
    """Compare Sparse TopK membership with boundary-equivalent fallback."""
    cpu = expected_indices.detach().cpu().to(torch.int32)
    npu = actual_indices.detach().cpu().to(torch.int32)
    report: dict[str, object] = {
        "pass": True,
        "mismatched_rows": [],
        "reason": "",
        "max_relative_error": 0.0,
    }
    if cpu.shape != npu.shape:
        report.update({"pass": False, "reason": "shape_mismatch"})
        return report
    flat_cpu = cpu.reshape(-1, cpu.shape[-1])
    flat_npu = npu.reshape(-1, npu.shape[-1])
    flat_cpu_values = None
    flat_npu_values = None
    if return_value:
        if (
            expected_values is None
            or actual_values is None
            or expected_values.shape != actual_values.shape
        ):
            report.update({"pass": False, "reason": "return_value_shape_mismatch"})
            return report
        flat_cpu_values = (
            expected_values.detach().cpu().reshape(-1, expected_values.shape[-1])
        )
        flat_npu_values = (
            actual_values.detach().cpu().reshape(-1, actual_values.shape[-1])
        )
    valid_lens = (flat_cpu != -1).sum(dim=-1)
    diff_row_indices: list[int] = []
    for begin in range(0, flat_cpu.shape[0], COMPARE_ROW_CHUNK):
        end = min(begin + COMPARE_ROW_CHUNK, flat_cpu.shape[0])
        cpu_sorted = torch.sort(flat_cpu[begin:end], dim=-1).values
        npu_sorted = torch.sort(flat_npu[begin:end], dim=-1).values
        local_rows = torch.nonzero(
            torch.any(cpu_sorted != npu_sorted, dim=-1), as_tuple=False
        ).flatten()
        diff_row_indices.extend(begin + int(row) for row in local_rows.tolist())
    mismatched_rows: list[int] = []
    max_re = 0.0
    for row in diff_row_indices:
        valid_len = int(valid_lens[row])
        try:
            score_row = _reference_row(reference_scores, row, candidate=False)
        except (IndexError, TypeError, ValueError):
            mismatched_rows.append(int(row))
            continue
        row_pass, row_re = _compare_topk_row(
            flat_cpu[row, :valid_len],
            flat_npu[row, :valid_len],
            score_row,
            cpu_output_row=None if flat_cpu_values is None else flat_cpu_values[row],
            npu_output_row=None if flat_npu_values is None else flat_npu_values[row],
            return_value=return_value,
        )
        max_re = max(max_re, row_re)
        if not row_pass:
            mismatched_rows.append(int(row))
    if mismatched_rows:
        report.update(
            {
                "pass": False,
                "reason": "row_mismatch",
                "mismatched_rows": mismatched_rows,
            }
        )
    report["max_relative_error"] = max_re
    return report


def compare_candidate_outputs(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    expected_scores=None,
) -> dict[str, object]:
    """Apply the QLI TopK rule to Candidate plus its full output contract."""
    cpu = expected["candidate_block_indices"].detach().cpu().to(torch.int32)
    npu = actual["candidate_block_indices"].detach().cpu().to(torch.int32)
    cpu_len = expected["candidate_block_length"].detach().cpu().to(torch.int64)
    npu_len = actual["candidate_block_length"].detach().cpu().to(torch.int64)
    scores = expected_scores
    if scores is None:
        scores = expected.get("_candidate_block_scores")
    report: dict[str, object] = {
        "pass": True,
        "mismatched_rows": [],
        "reason": "",
        "max_relative_error": 0.0,
    }
    if cpu.shape != npu.shape or cpu_len.shape != npu_len.shape:
        report.update({"pass": False, "reason": "shape_mismatch"})
        return report
    if not torch.equal(cpu_len, npu_len):
        flat_cpu_len = cpu_len.reshape(-1)
        flat_npu_len = npu_len.reshape(-1)
        rows = torch.nonzero(flat_cpu_len != flat_npu_len, as_tuple=False).flatten()
        sample_rows = rows[:16].tolist()
        report.update(
            {
                "pass": False,
                "reason": "length_mismatch",
                "mismatched_rows": [int(row) for row in rows.tolist()],
                "length_samples": [
                    {
                        "row": int(row),
                        "expected": int(flat_cpu_len[row]),
                        "actual": int(flat_npu_len[row]),
                    }
                    for row in sample_rows
                ],
            }
        )
        return report
    if cpu.shape[-1] == 0:
        if torch.any(cpu_len != 0):
            report.update(
                {"pass": False, "reason": "nonzero_disabled_candidate_length"}
            )
        return report

    flat_cpu = cpu.reshape(-1, cpu.shape[-1])
    flat_npu = npu.reshape(-1, npu.shape[-1])
    flat_len = cpu_len.reshape(-1)
    contract_rows: set[int] = set()
    set_diff_rows: list[int] = []
    width = flat_cpu.shape[-1]
    positions = torch.arange(width, dtype=torch.int64).reshape(1, width)
    sentinel = torch.iinfo(torch.int32).max
    for begin in range(0, flat_cpu.shape[0], COMPARE_ROW_CHUNK):
        end = min(begin + COMPARE_ROW_CHUNK, flat_cpu.shape[0])
        chunk_len = flat_len[begin:end].reshape(-1, 1)
        valid = positions < chunk_len
        cpu_chunk = flat_cpu[begin:end]
        npu_chunk = flat_npu[begin:end]
        bad = (
            ((chunk_len < 0) | (chunk_len > width)).reshape(-1)
            | torch.any(torch.where(valid, cpu_chunk < 0, cpu_chunk != -1), dim=-1)
            | torch.any(torch.where(valid, npu_chunk < 0, npu_chunk != -1), dim=-1)
        )
        cpu_normalized = torch.where(valid, cpu_chunk, sentinel)
        npu_normalized = torch.where(valid, npu_chunk, sentinel)
        cpu_sorted = torch.sort(cpu_normalized, dim=-1).values
        npu_sorted = torch.sort(npu_normalized, dim=-1).values
        if width > 1:
            valid_pairs = positions[:, :-1] < (chunk_len - 1)
            bad = (
                bad
                | torch.any(
                    (cpu_sorted[:, :-1] == cpu_sorted[:, 1:]) & valid_pairs,
                    dim=-1,
                )
                | torch.any(
                    (npu_sorted[:, :-1] == npu_sorted[:, 1:]) & valid_pairs,
                    dim=-1,
                )
            )
        different = torch.any(cpu_sorted != npu_sorted, dim=-1)
        for local_row in torch.nonzero(bad, as_tuple=False).flatten().tolist():
            contract_rows.add(begin + int(local_row))
        for local_row in (
            torch.nonzero(different & ~bad, as_tuple=False).flatten().tolist()
        ):
            set_diff_rows.append(begin + int(local_row))

    mismatched_rows: list[int] = sorted(contract_rows)
    max_re = 0.0
    for row in set_diff_rows:
        valid_len = int(flat_len[row])
        cpu_valid = flat_cpu[row, :valid_len]
        npu_valid = flat_npu[row, :valid_len]
        if scores is None:
            mismatched_rows.append(row)
            continue
        try:
            score_row = _reference_row(scores, row, candidate=True)
        except (IndexError, TypeError, ValueError):
            mismatched_rows.append(row)
            continue
        row_pass, row_re = _compare_topk_row(
            cpu_valid,
            npu_valid,
            score_row,
            cpu_output_row=score_row.index_select(0, cpu_valid),
            npu_output_row=score_row.index_select(0, npu_valid),
            return_value=True,
        )
        max_re = max(max_re, row_re)
        if not row_pass:
            mismatched_rows.append(row)
    if mismatched_rows:
        report.update(
            {
                "pass": False,
                "reason": "row_mismatch",
                "mismatched_rows": mismatched_rows,
            }
        )
    report["max_relative_error"] = max_re
    return report


def compare_sparse_outputs(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Compare Sparse output with the QLI ResultCompare policy."""
    reference_scores = expected.get("_reference_scores")
    if reference_scores is None:
        return {
            "pass": False,
            "mismatched_rows": [],
            "reason": "missing_reference_scores",
        }
    return_value = bool(expected.get("_return_value", False))
    expected_indices = expected["sparse_indices"]
    actual_indices = actual["sparse_indices"]
    offsets = expected.get("_output_idx_offset")
    if offsets is not None:
        offsets = offsets.unsqueeze(-1)
        expected_indices = torch.where(
            expected_indices >= 0, expected_indices - offsets, expected_indices
        )
        actual_indices = torch.where(
            actual_indices >= 0, actual_indices - offsets, actual_indices
        )
    return _compare_sparse_indices(
        expected_indices,
        actual_indices,
        reference_scores,
        expected.get("sparse_values"),
        actual.get("sparse_values"),
        return_value=return_value,
    )


def compare_qli_outputs(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Unified QLI precision comparison for Sparse TopK and Candidate outputs."""
    sparse = compare_sparse_outputs(expected, actual)
    candidate = compare_candidate_outputs(expected, actual)
    return {
        "pass": bool(sparse["pass"] and candidate["pass"]),
        "sparse": sparse,
        "candidate": candidate,
    }


__all__ = [
    "compare_candidate_outputs",
    "compare_qli_outputs",
    "compare_sparse_outputs",
]
