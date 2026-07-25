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

import logging
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "ttk", "qfa_mxfp4_test")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import result_compare_method

logger = logging.getLogger(__name__)


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


def compare(*outputs, **kwargs):
    non_none = [o for o in outputs if o is not None]

    if len(non_none) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs: {len(non_none)}",
        }

    all_empty = all(
        (hasattr(o, "size") and o.size == 0) or (hasattr(o, "numel") and o.numel() == 0)
        for o in non_none
    )
    if all_empty:
        logger.info("[COMPARE] 所有输出都是空 tensor, 视为 PASS")
        return {
            "pass": True,
            "precision": "100.0%",
            "error_info": None,
            "metrics": {"result": "Pass", "PctRlt": "100.0%", "MaxRE": 0.0},
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
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs after filtering: {len(valid)}",
        }

    half = len(valid) // 2
    npu_outputs = valid[:half]
    golden_outputs = valid[half:]

    all_pass = True
    results = []

    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        label = "atten" if idx == 0 else f"out{idx}"
        npu_torch = _to_torch(npu_out)
        golden_torch = _to_torch(golden_out)
        result, fulfill_percent, max_error = result_compare_method.check_result(
            golden_torch, npu_torch
        )
        is_pass = result == "Pass"
        if not is_pass:
            all_pass = False
        results.append(
            {
                "pass": is_pass,
                "precision": fulfill_percent,
                "error_info": None
                if is_pass
                else f"{label}: result={result}, MaxRE={max_error}",
                "metrics": {
                    "label": label,
                    "result": result,
                    "PctRlt": fulfill_percent,
                    "MaxRE": max_error,
                },
            }
        )

    if len(results) == 1:
        return results[0]
    return results
