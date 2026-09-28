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
from sparse_flash_attention_grad_paramset import ENABLED_PARAMS
import sparse_flash_attention_grad_golden
from batch import sparse_flash_attention_grad_process
import result_compare_method
import utils
import os
import pytest

save_pt = False
pt_save_path = "data"

result_path = os.getenv("SFAG_RESULT_SAVE_PATH", "./result/sfag_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_sparse_flash_attention_grad(params):
    torch_npu.npu.set_device(0)
    layout = params[0]
    case_name = params[12]

    input_data = sparse_flash_attention_grad_golden.gen_data(params)
    if save_pt:
        sparse_flash_attention_grad_golden.save_test_case(input_data, pt_save_path)

    npu_results = sparse_flash_attention_grad_process.call_npu(input_data)

    npu_dq = npu_results[0]
    npu_dk = npu_results[1]
    npu_dv = npu_results[2]
    npu_dq_rope = npu_results[3] if len(npu_results) > 3 else None
    npu_dk_rope = npu_results[4] if len(npu_results) > 4 else None

    result, fulfill_percent = result_compare_method.check_result(
        input_data["cpu_dq"], npu_dq
    )
    result_dk, _ = result_compare_method.check_result(input_data["cpu_dk"], npu_dk)
    if result_dk != "Pass":
        result = "Failed"
    result_dv, _ = result_compare_method.check_result(input_data["cpu_dv"], npu_dv)
    if result_dv != "Pass":
        result = "Failed"
    if npu_dq_rope is not None:
        result_dq_rope, _ = result_compare_method.check_result(
            input_data["cpu_dq_rope"], npu_dq_rope
        )
        if result_dq_rope != "Pass":
            result = "Failed"
    if npu_dk_rope is not None:
        result_dk_rope, _ = result_compare_method.check_result(
            input_data["cpu_dk_rope"], npu_dk_rope
        )
        if result_dk_rope != "Pass":
            result = "Failed"

    utils.save_result(result, fulfill_percent, params, result_path)
    assert result == "Pass", f"test case {case_name} (layout={layout}) failed"
