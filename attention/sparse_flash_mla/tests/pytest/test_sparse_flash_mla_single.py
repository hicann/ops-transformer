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

import result_compare_method
import utils
import torch_npu
from sparse_flash_mla_paramset import ENABLED_PARAMS
import sparse_flash_mla_golden
import os
import pytest
from batch import sparse_flash_mla_process

save_pt = False
pt_save_path = "data"

# 处理所有参数组合
result_path = os.getenv("SMLA_RESULT_SAVE_PATH", "./result/smla_result_all_sparse.xlsx")
param_combinations = utils.generate_param_combinations(ENABLED_PARAMS)


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_example(param_combinations):
    test_data = utils.generate_case_with_default_param(param_combinations)
    torch_npu.npu.set_device(0)

    print("test_data:", test_data)
    # 获得cpu结果(真值)和算子结果（测试值）
    input_data = sparse_flash_mla_golden.gen_data(test_data)
    if save_pt:
        sparse_flash_mla_golden.save_test_case(input_data, pt_save_path)
    npu_error_msg = None
    try:
        npu_result, softmax_lse = sparse_flash_mla_process.call_npu(input_data)
    except Exception as e:
        npu_error_msg = str(e)
        print("NPU ERROR: ", npu_error_msg)
        npu_result = None

    if npu_error_msg is not None:
        result = "NPU ERROR"
        fulfill_percent = 0
    else:
        print("npu_result.size():", npu_result.size())
        # 结果精度对比
        result, fulfill_percent = result_compare_method.check_result(
            input_data["cpu_output"], npu_result
        )

    if test_data.get("return_softmax_lse", False):
        result, fulfill_percent = result_compare_method.check_result(
            input_data["softmax_lse"], softmax_lse
        )
    # 记录结果
    utils.save_result(result, fulfill_percent, test_data, result_path)

    if result == "Failed":
        pytest.fail(f"用例精度失败:{param_combinations} 精度:{fulfill_percent:.2f}%")
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败:{param_combinations} NPU ERROR: {npu_error_msg}")
