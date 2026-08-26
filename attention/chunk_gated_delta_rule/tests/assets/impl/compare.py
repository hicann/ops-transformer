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

"""Three-party cross_check compare for ChunkGatedDeltaRule.

Compares NPU outputs (t) against CPU golden (g) and benchmark (b).
Pass/fail mirrors pytest compare_cv: NPU error ratios relative to benchmark
error must stay within thresholds (mare/mere/rmse + small-value error ratio).
"""

import logging

logger = logging.getLogger(__name__)
import datetime
import os
import sys

import numpy as np
import torch

_MIN_ERR = 1e-3
_ERR_THRESHOLD = 2 ** (-8)
_SMALL_VALUE = 2 ** (-10)
_SMALL_VALUE_ATOL = 1e-16

CV_MAX_RE = 5.0
CV_AVER_RE = 1.5
CV_RMSE = 1.5
CV_SMALL_VAL = 2.0

_OUTPUT_NAMES = ["out", "finalState"]


def _print_log(data):
    logger.info(
        "[%s] [INFO]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
            value = value.to(torch.float32)
        return value.numpy()
    return np.asarray(value)


def _get_max_re(actual, golden):
    abs_error = np.abs(actual - golden) / (np.abs(golden) + _MIN_ERR)
    return float(np.max(abs_error))


def _get_avg_re(actual, golden):
    abs_error = np.abs(actual - golden) / (np.abs(golden) + _MIN_ERR)
    return float(np.mean(abs_error))


def _get_rmse(actual, golden):
    sqr_err = np.power(actual - golden, 2)
    return float(np.sqrt(np.mean(sqr_err)))


def _get_smra(actual, golden):
    abs_g = np.abs(golden)
    mask_small = abs_g < _SMALL_VALUE
    num_small = int(np.sum(mask_small))
    if num_small == 0:
        return 0.0, 0, 0
    mask_err = np.abs(golden - actual) > _SMALL_VALUE_ATOL
    num_err = int(np.sum(mask_small & mask_err))
    return (num_err / num_small if num_small > 0 else 0.0), num_err, num_small


def _safe_div(num, den):
    if den is None or den == 0:
        return 1.0 if num == 0 else float("inf")
    return float(num / max(den, _ERR_THRESHOLD))


def cross_check_output(npu_out, golden_out, bench_out, name):
    """Three-party comparison for a single output.

    Computes mare/mere/rmse/smra ratios (NPU error / benchmark error) and
    checks against thresholds (CV_MAX_RE / CV_AVER_RE / CV_RMSE / CV_SMALL_VAL).
    """
    g = as_numpy(golden_out).reshape(-1).astype(np.float32)
    t = as_numpy(npu_out).reshape(-1).astype(np.float32)

    if t.shape != g.shape:
        result = {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"{name} shape mismatch: npu={t.shape}, golden={g.shape}",
        }
        del t, g
        return result

    max_re_npu = _get_max_re(t, g)
    avg_re_npu = _get_avg_re(t, g)
    rmse_npu = _get_rmse(t, g)
    smra_npu, err_npu, num_small = _get_smra(t, g)

    del t

    b = as_numpy(bench_out).reshape(-1).astype(np.float32)

    if b.shape != g.shape:
        result = {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"{name} shape mismatch: bench={b.shape}, golden={g.shape}",
        }
        del b, g
        return result

    max_re_bench = _get_max_re(b, g)
    avg_re_bench = _get_avg_re(b, g)
    rmse_bench = _get_rmse(b, g)
    smra_bench, err_bench, _ = _get_smra(b, g)

    del b, g

    max_re_rate = _safe_div(max_re_npu, max_re_bench)
    avg_re_rate = _safe_div(avg_re_npu, avg_re_bench)
    rmse_rate = _safe_div(rmse_npu, rmse_bench)
    smra_rate = _safe_div(smra_npu, smra_bench)

    _print_log(
        f"---------------------------------------cross_check {name}------------------------------------------------"
    )
    _print_log(
        "max_re_rate=%.3f (%.1f), max_re_bench=%.3e | "
        "avg_re_rate=%.3f (%.1f), avg_re_bench=%.3e | "
        "rmse_rate=%.3f (%.1f), rmse_bench=%.3e | "
        "smra_rate=%.3f (%.1f)"
        % (
            max_re_rate,
            CV_MAX_RE,
            max_re_bench,
            avg_re_rate,
            CV_AVER_RE,
            avg_re_bench,
            rmse_rate,
            CV_RMSE,
            rmse_bench,
            smra_rate,
            CV_SMALL_VAL,
        )
    )

    passed = (
        max_re_rate < CV_MAX_RE
        and avg_re_rate < CV_AVER_RE
        and rmse_rate < CV_RMSE
        and smra_rate < CV_SMALL_VAL
    )

    if not passed:
        epsilon = 2.0 ** (-7)
        if max_re_npu < epsilon:
            _print_log(f"\tmax_re_npu={max_re_npu} less than {epsilon}.")
            passed = True

    result = "Pass" if passed else "Failed"
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log(f"{name}: {result}")

    error_info = None
    if not passed:
        exceeded = []
        if max_re_rate >= CV_MAX_RE:
            exceeded.append(f"mare({max_re_rate:.2f}>{CV_MAX_RE})")
        if avg_re_rate >= CV_AVER_RE:
            exceeded.append(f"mere({avg_re_rate:.2f}>{CV_AVER_RE})")
        if rmse_rate >= CV_RMSE:
            exceeded.append(f"rmse({rmse_rate:.2f}>{CV_RMSE})")
        if smra_rate >= CV_SMALL_VAL:
            exceeded.append(f"smra({smra_rate:.2f}>{CV_SMALL_VAL})")
        error_info = f"{name} cross_check failed: {', '.join(exceeded)}"

    precision = 100.0 if passed else (100.0 / max(max_re_rate, 1.0))

    return {
        "pass": passed,
        "precision": precision,
        "error_info": error_info,
        "metrics": {
            "standard": "cross_check",
            "name": name,
            "mare_ratio": max_re_rate,
            "mere_ratio": avg_re_rate,
            "rmse_ratio": rmse_rate,
            "smra_ratio": smra_rate,
            "max_re_npu": max_re_npu,
            "max_re_bench": max_re_bench,
            "avg_re_npu": avg_re_npu,
            "avg_re_bench": avg_re_bench,
            "rmse_npu": rmse_npu,
            "rmse_bench": rmse_bench,
            "small_err_npu": err_npu,
            "small_err_bench": err_bench,
            "small_count": num_small,
        },
    }


def compare(*outputs, **kwargs):
    """Three-party compare: NPU vs golden vs benchmark.

    Receives NPU outputs followed by golden outputs (same count).
    Benchmark outputs are passed via kwargs (bench_out, bench_state).
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }

    bench_out = kwargs.get("bench_out")
    bench_state = kwargs.get("bench_state")
    bench_outputs = [bench_out, bench_state]

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    results = []
    for i in range(half):
        name = _OUTPUT_NAMES[i] if i < len(_OUTPUT_NAMES) else f"output_{i}"
        b_out = bench_outputs[i] if i < len(bench_outputs) else None
        if b_out is None:
            results.append(
                {
                    "pass": False,
                    "precision": "N/A",
                    "error_info": f"{name}: benchmark output not available",
                }
            )
            continue
        results.append(
            cross_check_output(npu_outputs[i], golden_outputs[i], b_out, name)
        )

    return results[0] if len(results) == 1 else results
