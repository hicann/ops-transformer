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

import concurrent.futures

import pytest

from common import test_runner
import fia_fullquant_mxfp8_paramset_perf_rdv as paramset

CASES = paramset.CASES
SKIP_CASES = getattr(paramset, "SKIP_CASES", set())
CASE_IDS = [case["name"] for case in CASES]


def _build_parametrize_args():
    args = []
    for case in CASES:
        if case["name"] in SKIP_CASES:
            args.append(pytest.param(case, marks=pytest.mark.skip(reason="execution time too long")))
        else:
            args.append(case)
    return args


@pytest.mark.perf_rdv
@pytest.mark.parametrize("params", _build_parametrize_args(), ids=CASE_IDS)
def test_fia_fullquant_mxfp8(params, golden_mode, cache_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_runner.execute_test, params, golden_mode, cache_dir)
        atten_result, lse_result = future.result()
    test_runner.check_results(atten_result, lse_result)
