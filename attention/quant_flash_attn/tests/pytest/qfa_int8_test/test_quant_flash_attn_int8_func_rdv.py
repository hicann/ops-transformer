# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest

from .common import test_runner
from .quant_flash_attn_int8_paramset_func_rdv import CASES


@pytest.mark.func_rdv
@pytest.mark.ci
@pytest.mark.parametrize("params", CASES, ids=[case["name"] for case in CASES])
def test_quant_flash_attn_gqa(params, golden_mode, cache_dir):
    test_runner.check_results(*test_runner.execute_test(params, golden_mode, cache_dir))
