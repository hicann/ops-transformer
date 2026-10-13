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

"""Precision comparison aligned with the FlashMlaWithKvcache pytest policy."""

import numpy as np


class FlashMlaWithKvcacheComparator:
    """Apply the pytest relative-error and global failure-ratio policy."""

    DEFAULT_RTOL = 0.005
    BFLOAT16_RTOL = 0.0078125
    DEFAULT_ATOL = 0.000025
    BFLOAT16_ATOL = 0.0001
    FAIL_RATIO = 0.005
    RELATIVE_FLOOR = (1.0 / (1 << 14)) / 0.005
    RELATIVE_EPSILON = 2e-9

    @staticmethod
    def is_bfloat16(value):
        return "bfloat16" in str(getattr(value, "dtype", ""))

    @classmethod
    def as_float32(cls, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if cls.is_bfloat16(value):
                value = value.float()
            value = value.numpy()
        return np.asarray(value).astype(np.float32)

    @staticmethod
    def print_log(message, output_index):
        """Print diagnostics directly, including when used outside TTK."""
        for line in message.splitlines():
            print(f"[INFO] [Output {output_index}] {line}", flush=True)

    @staticmethod
    def output_name(output_index):
        return {0: "ATTENTION", 1: "LSE"}.get(output_index, f"OUTPUT {output_index}")

    @staticmethod
    def table_header():
        header = (
            f"{'Loop':>8}  "
            f"{'ExpectOut':>14}  {'RealOut':>14}  {'FpDiff':>14}  {'RateDiff':>14}"
        )
        return header, "-" * len(header)

    @staticmethod
    def format_point(index, expected, actual, absolute, relative):
        return (
            f"{index:08d}  "
            f"{expected:14.7f}  {actual:14.7f}  {absolute:14.7e}  {relative:14.7e}"
        )

    @classmethod
    def display_samples(cls, npu, golden, precision, output_index, *, passed):
        """Print bounded first/last samples regardless of comparison status."""
        header, separator = cls.table_header()
        lines = [
            f"{cls.output_name(output_index)} {'PASS' if passed else 'FAIL'} "
            f"| Output samples: precision={precision:.6g}%, "
            f"shape={npu.shape}, elements={npu.size}",
            separator,
            header,
            separator,
        ]
        indices = (
            list(range(npu.size))
            if npu.size <= 20
            else [*range(10), *range(npu.size - 10, npu.size)]
        )
        actual, expected = npu.reshape(-1), golden.reshape(-1)
        for position, index in enumerate(indices):
            if npu.size > 20 and position == 10:
                lines.append("...")
            a, e = float(actual[index]), float(expected[index])
            absolute = abs(a - e)
            relative = absolute / (
                max(abs(a), abs(e), cls.RELATIVE_FLOOR) + cls.RELATIVE_EPSILON
            )
            lines.append(cls.format_point(index, e, a, absolute, relative))
        lines.append(separator)
        cls.print_log("\n".join(lines), output_index)

    @classmethod
    def display_error_output(
        cls, npu, golden, diff_idx, output_index, *, passed, precision
    ):
        """Print bounded mismatch points with the overall comparison status."""
        if not diff_idx.size:
            return ""
        actual = npu.reshape(-1)[diff_idx].astype(np.float64)
        expected = golden.reshape(-1)[diff_idx].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            absolute = np.abs(actual - expected)
            denominator = np.maximum(
                np.maximum(np.abs(actual), np.abs(expected)), cls.RELATIVE_FLOOR
            )
            relative = absolute / (denominator + cls.RELATIVE_EPSILON)

        def point(position):
            flat_index = int(diff_idx[position])
            return cls.format_point(
                flat_index,
                expected[position],
                actual[position],
                absolute[position],
                relative[position],
            )

        header, separator = cls.table_header()
        lines = [
            f"{cls.output_name(output_index)} {'PASS' if passed else 'FAIL'} "
            f"| Failed points: precision={precision:.6g}%, "
            f"shape={npu.shape}, elements={npu.size}",
            f"Error Line: {diff_idx.size} mismatches (first 10 and last 10 shown; no duplicates)",
            separator,
            header,
            separator,
        ]
        positions = (
            list(range(diff_idx.size))
            if diff_idx.size <= 20
            else [*range(10), *range(diff_idx.size - 10, diff_idx.size)]
        )
        for count, position in enumerate(positions):
            if diff_idx.size > 20 and count == 10:
                lines.append("...")
            lines.append(point(position))
        lines.append(separator)
        cls.print_log("\n".join(lines), output_index)

    @classmethod
    def compare_output(cls, npu_out, golden_out, output_index=0):
        if golden_out is None:
            cls.print_log(
                f"{cls.output_name(output_index)} SUPPRESSED: "
                "golden output is None; comparison skipped",
                output_index,
            )
            return {"pass": True, "precision": "SUPPRESSED"}
        if npu_out is None:
            cls.print_log(
                f"{cls.output_name(output_index)} FAIL: NPU output is None",
                output_index,
            )
            return {
                "pass": False,
                "precision": "NO_OUTPUT",
                "error_info": "NPU output is None",
            }

        is_bfloat16 = cls.is_bfloat16(npu_out)
        rtol = cls.BFLOAT16_RTOL if is_bfloat16 else cls.DEFAULT_RTOL
        atol = cls.BFLOAT16_ATOL if is_bfloat16 else cls.DEFAULT_ATOL
        npu = cls.as_float32(npu_out)
        golden = cls.as_float32(golden_out)
        if npu.shape != golden.shape:
            cls.print_log(
                f"{cls.output_name(output_index)} FAIL: "
                f"output shape mismatch: npu={npu.shape}, golden={golden.shape}",
                output_index,
            )
            return {
                "pass": False,
                "precision": "shape_mismatch",
                "error_info": f"output shape mismatch: npu={npu.shape}, golden={golden.shape}",
            }
        if golden.size == 0:
            cls.display_samples(npu, golden, 100.0, output_index, passed=True)
            return {"pass": True, "precision": 100.0}

        npu_flat = npu.reshape(-1)
        golden_flat = golden.reshape(-1)
        mismatch = ~np.isclose(
            npu_flat, golden_flat, rtol=rtol, atol=atol, equal_nan=True
        )
        diff_idx = np.where(mismatch)[0]
        fail_ratio = diff_idx.size / golden_flat.size
        max_relative_error = 0.0
        if diff_idx.size:
            diff_abs = np.abs(golden_flat - npu_flat)
            denominator = np.maximum(
                np.maximum(np.abs(npu_flat), np.abs(golden_flat)), cls.RELATIVE_FLOOR
            )
            relative_error = diff_abs / (denominator + cls.RELATIVE_EPSILON)
            max_relative_error = float(np.max(relative_error[diff_idx]))

        nonfinite_mismatch = mismatch & (
            ~np.isfinite(npu_flat) | ~np.isfinite(golden_flat)
        )
        nonfinite_mismatch_count = int(np.count_nonzero(nonfinite_mismatch))
        passed = fail_ratio <= cls.FAIL_RATIO and nonfinite_mismatch_count == 0
        precision = (golden_flat.size - diff_idx.size) / golden_flat.size * 100
        error_info = None
        if not passed:
            error_info = (
                f"FlashMlaWithKvcache precision failed: mismatches={diff_idx.size}, "
                f"fail_ratio={fail_ratio:.6g}, "
                f"nonfinite_mismatches={nonfinite_mismatch_count}, "
                f"max_relative_error={max_relative_error:.6g}"
            )
            cls.print_log(error_info, output_index)
        cls.display_samples(npu, golden, precision, output_index, passed=passed)
        if not passed and diff_idx.size:
            cls.display_error_output(
                npu, golden, diff_idx, output_index, passed=passed, precision=precision
            )
        return {
            "pass": passed,
            "precision": precision,
            "diff_indices": diff_idx[:1000].tolist(),
            "error_info": error_info,
            "metrics": {
                "rtol": rtol,
                "atol": atol,
                "fail_ratio": fail_ratio,
                "fail_ratio_limit": cls.FAIL_RATIO,
                "nonfinite_mismatch_count": nonfinite_mismatch_count,
                "max_relative_error": max_relative_error,
            },
        }


COMPARATOR = FlashMlaWithKvcacheComparator()


def compare(*outputs):
    """Compare NPU outputs followed by golden outputs."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    results = []
    for index, (npu_out, golden_out) in enumerate(zip(outputs[:half], outputs[half:])):
        if index != 1 or golden_out is None or npu_out is None:
            results.append(COMPARATOR.compare_output(npu_out, golden_out, index))
            continue
        npu = COMPARATOR.as_float32(npu_out)
        golden = COMPARATOR.as_float32(golden_out)
        passed = npu.shape == golden.shape and bool(
            np.all(np.isclose(npu, golden, rtol=0.0, atol=0.1, equal_nan=True))
        )
        error_info = None
        diff_idx = np.empty(0, dtype=np.int64)
        if not passed:
            if npu.shape != golden.shape:
                error_info = (
                    f"MLA LSE shape mismatch: npu={npu.shape}, golden={golden.shape}"
                )
            else:
                diff_idx = np.flatnonzero(
                    ~np.isclose(npu, golden, rtol=0.0, atol=0.1, equal_nan=True)
                )
                error_info = (
                    "MLA LSE absolute error exceeds 0.1 or non-finite value mismatch: "
                    f"mismatches={diff_idx.size}"
                )
            COMPARATOR.print_log(f"LSE FAIL: {error_info}", index)
        precision = 100.0 if passed else 0.0
        if npu.shape == golden.shape:
            COMPARATOR.display_samples(npu, golden, precision, index, passed=passed)
        if diff_idx.size:
            COMPARATOR.display_error_output(
                npu, golden, diff_idx, index, passed=passed, precision=precision
            )
        results.append(
            {
                "pass": passed,
                "precision": precision,
                "error_info": error_info,
                "diff_indices": diff_idx[:1000].tolist(),
            }
        )
    return results
