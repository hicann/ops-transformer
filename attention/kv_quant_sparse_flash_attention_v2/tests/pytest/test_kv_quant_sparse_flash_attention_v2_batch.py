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

import os
from pathlib import Path
import concurrent.futures
import pytest
import torch
import utils


TESTCASE_PATH = os.environ.get("PT_FILES_PATH", "./pt_files/")
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0

locals()["testcase_files"] = []
if os.path.isfile(TESTCASE_PATH) and TESTCASE_PATH.endswith(".pt"):
    locals()["testcase_files"] = [TESTCASE_PATH]
    print(f"指定单个 pt 文件: {TESTCASE_PATH}")
elif os.path.isdir(TESTCASE_PATH):
    pt_files = [f for f in os.listdir(TESTCASE_PATH) if f.endswith(".pt")]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {TESTCASE_PATH}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(TESTCASE_PATH, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {TESTCASE_PATH}")


def execute_qsfa_v2(testcase_files):
    test_data = torch.load(testcase_files, map_location="cpu")
    testcase_name = os.path.basename(testcase_files).replace(".pt", "")
    result, fulfill_percent = utils.qsfa_v2_run_npu(
        test_data,
        testcase_name=testcase_name,
        device_id=DEVICE_ID,
        result_path=RESULT_PATH,
    )
    return result, fulfill_percent, test_data


@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_kv_quant_sparse_flash_attention_v2(testcase_files):
    test_data = None
    fulfill_percent = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa_v2, testcase_files)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result, fulfill_percent, test_data = future.result()
                if result == "Failed":
                    case_name = os.path.basename(testcase_files).replace(".pt", "")
                    pytest.fail(
                        f"用例名: {case_name}, 精度: {fulfill_percent:.4f}%",
                        pytrace=False,
                    )
            except Exception as e:
                params = test_data.get("params") if test_data else None
                case_name = os.path.basename(testcase_files).replace(".pt", "")
                if params:
                    utils.save_result(params, "Failed", "", RESULT_PATH)
                pytest.fail(f"用例名: {case_name}, 当前用例线程执行失败")
