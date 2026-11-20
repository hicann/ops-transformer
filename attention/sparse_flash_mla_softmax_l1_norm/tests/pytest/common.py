#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""Common definitions for SparseFlashMlaSoftmaxL1Norm pytest cases."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import torch


DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
ATOL = 2e-4
RTOL = 2e-4
BASE_SEED = 2026111200


@dataclass
class CalculationContext:
    input_case: Dict[str, Any]
    inputs: Optional[Dict[str, torch.Tensor]] = None
    golden: Optional[torch.Tensor] = None


def input_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported input dtype: {name}")


def check_result(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Return the fraction and maximum error of elements inside the tolerance."""
    if actual.shape != expected.shape:
        return 0.0, float("inf")
    diff = torch.abs(actual.float() - expected.float())
    limit = ATOL + RTOL * torch.abs(expected.float())
    return float(torch.mean((diff <= limit).float())), float(diff.max().item())
