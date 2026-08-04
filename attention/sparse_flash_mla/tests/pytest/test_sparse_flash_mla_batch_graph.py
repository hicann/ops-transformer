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
from batch import sparse_flash_mla_process
import pytest
import concurrent.futures
import pandas as pd
from pathlib import Path
import os

pt_dir = os.getenv("SMLA_PT_LOAD_PATH", "./data")
result_path = Path(os.getenv("SMLA_RESULT_SAVE_PATH", './result/smla_result.xlsx'))
batch_test_mode = int(os.environ.get("SMLA_BATCH_TEST_MODE", 0))
excel_path = os.environ.get("SMLA_EXCEL_PATH", os.path.join(os.path.dirname(__file__), "excel", "example.xlsx"))
excel_sheet = os.environ.get("SMLA_EXCEL_SHEET", "CSA")
device_id = int(os.environ.get("SMLA_DEVICE_ID", 0))

_single_case_path = os.environ.get("QSAS_TESTCASE_PATH", "").strip()

locals()["testcase_files"] = []
if _single_case_path:
    if not os.path.isfile(_single_case_path):
        print(f"错误: 环境变量 QSAS_TESTCASE_PATH 指定的用例文件不存在: {_single_case_path}")
    else:
        print(f"单用例隔离模式, 仅执行: {_single_case_path}")
        locals()["testcase_files"].append(_single_case_path)
elif os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    elif batch_test_mode == 1:
        df = pd.read_excel(excel_path, sheet_name=excel_sheet)
        target_names = [str(name) for name in df['testcase_name'].dropna().tolist() if str(name) != 'None']
        if not target_names:
            print(f"错误: 表格中没有有效的testcase_name: {excel_path} sheet: {excel_sheet}")
        else:
            print(f"从表格[{excel_sheet}]中读取到 {len(target_names)} 个目标用例名")
            for target_name in target_names:
                matched = [f for f in pt_files if target_name in f]
                if matched:
                    for f in matched:
                        filepath = os.path.join(pt_dir, f)
                        if filepath not in locals()["testcase_files"]:
                            locals()["testcase_files"].append(filepath)
                else:
                    print(f"警告: 用例名 '{target_name}' 未匹配到任何.pt文件")
            print(f"按表格筛选后共 {len(locals()['testcase_files'])} 个测试用例文件")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(pt_dir, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {pt_dir}")

print("files:", locals()["testcase_files"])

def smla_graph(testcase_files):
    test_data = torch.load(testcase_files, map_location="cpu")
    npu_error_msg = None
    try:
        npu_result, softmax_lse = sparse_flash_mla_process.call_npu_graph(test_data, device_id=device_id)
        attn_result, attn_percent = result_compare_method.check_result(test_data['cpu_output'], npu_result)
        lse_result, lse_percent = None, None
        fail_info = []
        min_fulfill = attn_percent
        if test_data['params'].get('return_softmax_lse'):
            print("return_softmax_lse is true!!!")
            lse_result, lse_percent = result_compare_method.check_result(test_data['softmax_lse'], softmax_lse)
            min_fulfill = min(min_fulfill, lse_percent)

        if attn_result != "Pass":
            fail_info.append(f"MAIN_FAILED:{attn_result}")
        if lse_result is not None and lse_result != "Pass":
            fail_info.append(f"LSE_FAILED:{lse_result}")

        if fail_info:
            result = "; ".join(fail_info)
            fulfill_percent = min_fulfill
        else:
            result = "Pass"
            fulfill_percent = min_fulfill
    except Exception as e:
        npu_error_msg = str(e)
        print("NPU ERROR：", npu_error_msg)
        result = "NPU ERROR"
        fulfill_percent = 0

    utils.save_result(result, fulfill_percent, test_data['params'], result_path)

    if result == "Failed":
        pytest.fail(f"用例精度失败:{os.path.basename(testcase_files)} 精度:{fulfill_percent:.2f}%")
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败:{os.path.basename(testcase_files)} NPU ERROR: {npu_error_msg}")

testcase_ids = [os.path.splitext(os.path.basename(f))[0] for f in locals()["testcase_files"]]

@pytest.mark.graph
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"], ids=testcase_ids)
def test_sparse_flash_mla(testcase_files):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(smla_graph, testcase_files)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")
