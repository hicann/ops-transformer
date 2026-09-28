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

import torch
import torch_npu
import result_compare_method
import utils
from batch import sparse_flash_attention_process
import pytest
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

pt_dir = os.getenv("SFA_PT_LOAD_PATH", "./data")
result_path = Path(os.getenv("SFA_RESULT_SAVE_PATH", "./result/sfa_result.xlsx"))

testcase_files = []
if os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            testcase_files.append(os.path.join(pt_dir, pt_file))
else:
    print(f"错误: 输出目录不存在: {pt_dir}")

print("files:", testcase_files)


def call_sfa_npu(testcase_file):
    print("执行文件: ", testcase_file)
    torch_npu.npu.set_device(0)
    test_data = torch.load(testcase_file, map_location="cpu")
    npu_result = None
    try:
        npu_result, npu_softmax_max, npu_softmax_sum = (
            sparse_flash_attention_process.call_npu(test_data)
        )
    except Exception as e:
        utils.save_result("Exception", 0, test_data["params"], result_path)
        raise e
    if npu_result is not None:
        result, fulfill_percent = result_compare_method.check_result(
            test_data["cpu_output"], npu_result
        )
        if test_data["attr"]["return_softmax_lse"]:
            result_max, _ = result_compare_method.check_result(
                test_data["cpu_softmax_max"], npu_softmax_max
            )
            result_sum, _ = result_compare_method.check_result(
                test_data["cpu_softmax_sum"], npu_softmax_sum
            )
            if result_max != "Pass" or result_sum != "Pass":
                result = "Failed"
    else:
        result = "Failed"
        fulfill_percent = 0
    utils.save_result(result, fulfill_percent, test_data["params"], result_path)


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", testcase_files)
def test_sparse_flash_attention(testcase_file):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_sfa_npu, testcase_file)
        for future in as_completed([future]):
            try:
                future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")
