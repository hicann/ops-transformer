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

"""pytest entry point for CPU/NPU SparseFlashMlaSoftmaxL1Norm comparison."""

import pytest
import torch

from common import CalculationContext, check_result
from cpu_impl import build_inputs, cpu_impl
from npu_impl import PTAtest
from test_case import TestCases

try:
    import torch_npu  # noqa: F401
except ImportError:  # Allows collection on a non-NPU development host.
    torch_npu = None


def test_npu_sparse_flash_mla_softmax_l1_norm():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch_npu/NPU is unavailable")

    for test_name, test_data in TestCases.items():
        case = dict(test_data)
        case["name"] = test_name
        ctx = CalculationContext(input_case=case)
        ctx.inputs = build_inputs(case)
        ctx.golden = cpu_impl(ctx)
        actual = PTAtest(ctx)
        ratio, max_abs = check_result(actual, ctx.golden)
        print(f"{test_name}: matched={ratio:.6f}, max_abs={max_abs:.9e}")
        assert ratio == 1.0, f"{test_name}: NPU result is outside tolerance"
