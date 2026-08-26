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
import fia_fullquant_mla_paramset_debug as paramset

CASES = paramset.CASES
CASE_IDS = [case["name"] for case in CASES]

# 给已知失败 case (paramset.FAIL_CASES) 打上 pytest.mark.fail 标记
# 使用: pytest -m fail 只跑失败 case; pytest -m "not fail" 只跑通过 case; pytest 全跑
_param_list = []
for _case, _cid in zip(CASES, CASE_IDS):
    if _cid in paramset.FAIL_CASES:
        _param_list.append(pytest.param(_case, id=_cid, marks=pytest.mark.fail))
    else:
        _param_list.append(pytest.param(_case, id=_cid))


@pytest.mark.debug
@pytest.mark.parametrize("params", _param_list)
def test_fia_fullquant_mla(params, golden_mode, cache_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            test_runner.execute_test, params, golden_mode, cache_dir
        )
        atten_result, lse_result = future.result()
    test_runner.check_results(atten_result, lse_result)
