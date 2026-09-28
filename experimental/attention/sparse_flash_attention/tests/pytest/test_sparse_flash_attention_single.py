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

import torch_npu
from sparse_flash_attention_paramset import ENABLED_PARAMS
import sparse_flash_attention_golden
from batch import sparse_flash_attention_process
import result_compare_method
import utils
import os
import pytest

save_pt = False
pt_save_path = "data"

result_path = os.getenv("SFA_RESULT_SAVE_PATH", "./result/sfa_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_sparse_flash_attention(params):
    torch_npu.npu.set_device(0)
    layout = params[0]
    return_softmax_lse = params[12]
    case_name = params[15]

    input_data = sparse_flash_attention_golden.gen_data(params)
    if save_pt:
        sparse_flash_attention_golden.save_test_case(input_data, pt_save_path)

    npu_result, npu_softmax_max, npu_softmax_sum = (
        sparse_flash_attention_process.call_npu(input_data)
    )

    result, fulfill_percent = result_compare_method.check_result(
        input_data["cpu_output"], npu_result
    )
    if return_softmax_lse:
        result_max, _ = result_compare_method.check_result(
            input_data["cpu_softmax_max"], npu_softmax_max
        )
        result_sum, _ = result_compare_method.check_result(
            input_data["cpu_softmax_sum"], npu_softmax_sum
        )
        if result_max != "Pass" or result_sum != "Pass":
            result = "Failed"

    utils.save_result(result, fulfill_percent, params, result_path)
    assert result == "Pass", f"test case {case_name} (layout={layout}) failed"
