#!/usr/bin/python3
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


"""TTK compare plugin for quant_flash_mla_with_kvcache.

Receives [npu_out(, npu_lse), golden_out(, golden_lse)] and checks
precision with bf16 tolerance defaults.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _to_torch(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.dtype.name == "bfloat16":
            return torch.from_numpy(x.view(np.uint8)).view(torch.bfloat16)
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def check_result(
    expect,
    npu_result,
    rtol=None,
    atol=None,
    diff_thd=None,
    pct_thd=None,
    max_diff_hd=None,
):
    _diff_thd_default = 0.005
    _pct_thd_default = 0.005
    _max_diff_hd_default = 10
    _rtol_default = 0.005
    _atol_default = 0.000025

    diff_thd = _diff_thd_default if diff_thd is None else diff_thd
    pct_thd = _pct_thd_default if pct_thd is None else pct_thd
    max_diff_hd = _max_diff_hd_default if max_diff_hd is None else max_diff_hd

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() == 0:
        return "Pass", 100.0, 0
    if real_data.numel() != expect_data.numel():
        return "Failed", 0.0, 0

    split_count = real_data.numel()
    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    if rtol is None or atol is None:
        if npu_result.dtype == torch.bfloat16:
            _rtol_default = 0.0078125
            _atol_default = 0.0001
        rtol = _rtol_default if rtol is None else rtol
        atol = _atol_default if atol is None else atol

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx] if err_idx.numel() > 0 else torch.tensor([])
    err_diff = torch.where(
        torch.isnan(err_diff) | torch.isinf(err_diff),
        torch.tensor(float("inf"), dtype=torch.float32),
        err_diff,
    )

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if fulfill_percent >= pct_thd else "Failed"

    max_error = 0.0
    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info(
        "Rtol=%.6f  Atol=%.6f  PctThd=%.2f%%  PctRlt=%.6f%%  Result=%s  MaxRE=%s",
        rtol,
        atol,
        pct_thd,
        fulfill_percent,
        result,
        max_error,
    )

    logger.info("First 100 values comparison (expect vs actual vs diff):")
    for i in range(min(100, real_data.numel())):
        e = float(expect_float[i].item())
        a = float(real_float[i].item())
        d = float(diff_abs[i].item())
        logger.info("  [%d] expect=%.6f actual=%.6f diff=%.6f", i, e, a, d)

    return result, fulfill_percent, max_error


def compare(*outputs, **kwargs):
    """Compare NPU outputs vs golden outputs.

    TTK passes [npu_out(, npu_lse), golden_out(, golden_lse)].
    Filters None and empty tensors, then pairs first half (NPU) with
    second half (golden).
    """
    non_none = [o for o in outputs if o is not None]
    if len(non_none) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs: {len(non_none)}",
        }

    valid = []
    for o in non_none:
        if hasattr(o, "size") and o.size == 0:
            continue
        if hasattr(o, "numel") and o.numel() == 0:
            continue
        valid.append(o)

    if len(valid) < 2:
        return {
            "pass": True,
            "precision": "100.0%",
            "error_info": None,
            "metrics": {"result": "Pass", "PctRlt": "100.0%", "MaxRE": 0.0},
        }

    half = len(valid) // 2
    npu_outputs = valid[:half]
    golden_outputs = valid[half:]

    results = []
    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        label = "atten" if idx == 0 else f"out{idx}"
        npu_torch = _to_torch(npu_out)
        golden_torch = _to_torch(golden_out)

        # bf16 defaults
        rtol, atol = 0.0078125, 0.0001
        if npu_torch.dtype != torch.bfloat16:
            rtol, atol = 0.005, 0.000025

        result, pct, max_err = check_result(
            golden_torch, npu_torch, rtol=rtol, atol=atol
        )
        results.append(
            {
                "pass": result == "Pass",
                "precision": pct,
                "error_info": None if result == "Pass" else f"{label}: MaxRE={max_err}",
                "metrics": {
                    "label": label,
                    "result": result,
                    "PctRlt": pct,
                    "MaxRE": max_err,
                },
            }
        )

    if len(results) == 1:
        return results[0]
    return results
