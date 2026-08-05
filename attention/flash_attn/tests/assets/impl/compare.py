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

"""Precision comparison aligned with the FlashAttn pytest policy."""

import numpy as np


class FlashAttnComparator:
    """Apply the pytest relative-error and global failure-ratio policy."""

    DEFAULT_RTOL = 0.005
    BFLOAT16_RTOL = 0.0078125
    DEFAULT_ATOL = 0.000025
    BFLOAT16_ATOL = 0.0001
    FAIL_RATIO = 0.005
    MAX_RELATIVE_ERROR = 10.0
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

    @classmethod
    def compare_output(cls, npu_out, golden_out):
        if golden_out is None:
            return {"pass": True, "precision": "SUPPRESSED"}
        if npu_out is None:
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
            return {
                "pass": False,
                "precision": "shape_mismatch",
                "error_info": f"output shape mismatch: npu={npu.shape}, golden={golden.shape}",
            }
        if golden.size == 0:
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

        passed = (
            fail_ratio <= cls.FAIL_RATIO and max_relative_error < cls.MAX_RELATIVE_ERROR
        )
        precision = (golden_flat.size - diff_idx.size) / golden_flat.size * 100
        error_info = None
        if not passed:
            error_info = (
                f"FlashAttn precision failed: mismatches={diff_idx.size}, "
                f"fail_ratio={fail_ratio:.6g}, "
                f"max_relative_error={max_relative_error:.6g}"
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
                "max_relative_error": max_relative_error,
                "max_relative_error_limit": cls.MAX_RELATIVE_ERROR,
            },
        }


COMPARATOR = FlashAttnComparator()


def compare(*outputs):
    """Compare NPU outputs followed by golden outputs."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    return [
        COMPARATOR.compare_output(npu_out, golden_out)
        for npu_out, golden_out in zip(outputs[:half], outputs[half:])
    ]
