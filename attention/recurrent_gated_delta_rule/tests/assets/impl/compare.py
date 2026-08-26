#!/usr/bin/env python3
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

"""Numerical compare helpers for RecurrentGatedDeltaRule TestSpec adapter.

Compares NPU outputs against golden references. Print format and pass/fail
criteria are kept consistent with tests/pytest/recurrent_gated_delta_rule_golden.py
check_result().
"""

import logging

logger = logging.getLogger(__name__)
import datetime
import os
import sys

import numpy as np
import torch

_RTOL = 0.0078125
_ATOL = 0.0001
_DIFF_THD = 0.005
_MAX_DIFF_HD = 10.0
_MAX_ERROR_IDX = 10000000
_DISPLAY_THRESHOLD = 1 << 20


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


def _cal_rel_diff(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (np.abs(expect_data) + 10e-10)


def _display_output_np_isclose(real_data, expect_data, start, end):
    def display_inner(idx):
        j = idx + start
        diff_rate = _cal_rel_diff(real_data[j], expect_data[j])
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            _print_log(
                "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                % (start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate)
            )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            _print_log(
                "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                % (start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate)
            )

    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        _print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def _display_error_output(real_data, expect_data, err_idx, relative_diff):
    _print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    _print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            _print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            _print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    _print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            _print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    _print_log(
        "---------------------------------------------------------------------------------------"
    )


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
            value = value.to(torch.float32)
        return value.numpy()
    return np.asarray(value)


def numerical_compare(npu_out, golden_out):
    """Compare a single NPU output against its golden reference.

    Print format and pass/fail criteria mirror pytest check_result().
    """
    npu = as_numpy(npu_out).reshape(-1).astype(np.float32, copy=False)
    golden = as_numpy(golden_out).reshape(-1).astype(np.float32, copy=False)
    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"shape mismatch: npu={npu.shape}, golden={golden.shape}",
        }
    total = npu.size
    if total == 0 and golden.size == 0:
        _print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return {"pass": True, "precision": 100.0}

    start = 0
    end = total - 1
    if end < start:
        end = start
    split_count = int(end - start + 1) if end != start else 1
    _print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), _MAX_DIFF_HD))

    eps = 10e-10
    b2 = float((1.0 / (1 << 14)) / _DIFF_THD)

    overflows_count = 0
    inf_samples = []
    nan_samples = []
    err_count = 0
    err_idx_parts = []
    err_diff_parts = []

    inf_mask_g = np.isinf(golden)
    nan_mask_g = np.isnan(golden)
    overflows_count = int(inf_mask_g.sum()) + int(nan_mask_g.sum())
    for li in np.where(inf_mask_g)[0]:
        if len(inf_samples) >= 10:
            break
        inf_samples.append(golden[li])
    for li in np.where(nan_mask_g)[0]:
        if len(nan_samples) >= 10:
            break
        nan_samples.append(golden[li])

    close = np.isclose(npu, golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    local_err = np.where(~close)[0]
    err_count = local_err.size
    if local_err.size > 0:
        er = npu[local_err]
        ec = golden[local_err]
        ed = np.abs(ec - er) / (
            np.maximum(np.maximum(np.abs(er), np.abs(ec)), b2) + eps + eps
        )
        err_idx_parts.append(local_err)
        err_diff_parts.append(ed)

    if overflows_count > 0:
        _print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                np.array(inf_samples)[0:10] if inf_samples else np.array([]),
                np.array(nan_samples)[0:10] if nan_samples else np.array([]),
            )
        )

    err_idx = (
        np.concatenate(err_idx_parts) if err_idx_parts else np.array([], dtype=np.int64)
    )
    err_diff = (
        np.concatenate(err_diff_parts)
        if err_diff_parts
        else np.array([], dtype=np.float64)
    )

    fulfill_percent = float(split_count - err_count) / float(split_count) * 100.0

    if total <= _DISPLAY_THRESHOLD:
        _display_output_np_isclose(npu, golden, start, end)
    else:
        _print_log(
            "---------------------------------------------------------------------------------------"
        )
        _print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
        _print_log(
            "---------------------------------------------------------------------------------------"
        )
        n_sample = 10
        for i in range(n_sample):
            diff_abs = abs(np.float64(golden[i]) - np.float64(npu[i]))
            diff_rate = _cal_rel_diff(npu[i], golden[i])
            _print_log(
                "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                % (i + 1, golden[i], npu[i], diff_abs, diff_rate)
            )
        _print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(n_sample):
            j = total - n_sample + i
            diff_abs = abs(np.float64(golden[j]) - np.float64(npu[j]))
            diff_rate = _cal_rel_diff(npu[j], golden[j])
            _print_log(
                "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                % (j + 1, golden[j], npu[j], diff_abs, diff_rate)
            )

    pct_thd = (1 - _DIFF_THD) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    max_error = 0.0
    if len(err_diff) > 0:
        max_error = max(err_diff[0:_MAX_ERROR_IDX])
        if max_error >= _MAX_DIFF_HD:
            result = "Failed"
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (_RTOL, _ATOL, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        _print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, _MAX_DIFF_HD)
        )
    if result == "Failed":
        err_limit = min(len(err_idx), _MAX_ERROR_IDX)
        if err_limit > 0:
            err_indices = err_idx[:err_limit]
            err_r = npu[err_indices]
            err_c = golden[err_indices]
            _display_error_output(
                err_r, err_c, np.arange(err_limit), err_diff[:err_limit]
            )

    passed = result == "Pass"
    error_info = None
    if not passed:
        error_info = (
            f"fulfill_percent={fulfill_percent:.6f}%, max_rel_err={max_error:.6f}, "
            f"err_count={err_count}/{split_count}"
        )
    return {
        "pass": passed,
        "precision": fulfill_percent,
        "diff_indices": err_idx[:1000].tolist(),
        "error_info": error_info,
        "metrics": {
            "rtol": _RTOL,
            "atol": _ATOL,
            "pct_thd": pct_thd,
            "fulfill_percent": fulfill_percent,
            "max_rel_err": max_error,
            "max_diff_hd": _MAX_DIFF_HD,
        },
    }


def compare(*outputs, **kwargs):
    """Compare NPU outputs against golden references.

    Receives NPU outputs followed by golden outputs (same count). Returns a dict
    (single output) or list of dicts (multiple outputs).
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    results = [
        numerical_compare(npu_outputs[i], golden_outputs[i]) for i in range(half)
    ]
    return results[0] if len(results) == 1 else results
