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
from batch_consistency.config import resolve_consistency_config
from batch_consistency.model import RunResult
from batch_consistency.pytest_support import (
    consistency_fulfill_percent,
    format_consistency_summary,
    run_configured_consistency,
)
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import os

pt_dir = os.getenv("SMLA_PT_LOAD_PATH", "./data")
result_path = Path(os.getenv("SMLA_RESULT_SAVE_PATH", "./result/smla_result.xlsx"))
batch_test_mode = int(os.environ.get("SMLA_BATCH_TEST_MODE", 0))
excel_path = os.environ.get(
    "SMLA_EXCEL_PATH", os.path.join(os.path.dirname(__file__), "excel", "example.xlsx")
)
excel_sheet = os.environ.get("SMLA_EXCEL_SHEET", "CSA")

_single_case_path = os.environ.get("QSAS_TESTCASE_PATH", "").strip()

locals()["testcase_files"] = []
if _single_case_path:
    if not os.path.isfile(_single_case_path):
        print(
            f"错误: 环境变量 QSAS_TESTCASE_PATH 指定的用例文件不存在: {_single_case_path}"
        )
    else:
        print(f"单用例隔离模式, 仅执行: {_single_case_path}")
        locals()["testcase_files"].append(_single_case_path)
elif os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    elif batch_test_mode == 1:
        df = pd.read_excel(excel_path, sheet_name=excel_sheet)
        target_names = [
            str(name)
            for name in df["testcase_name"].dropna().tolist()
            if str(name) != "None"
        ]
        if not target_names:
            print(
                f"错误: 表格中没有有效的testcase_name: {excel_path} sheet: {excel_sheet}"
            )
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


@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_sparse_flash_mla(testcase_files):
    print("执行文件: ", testcase_files)
    torch_npu.npu.set_device(0)
    test_data = torch.load(testcase_files, map_location="cpu", weights_only=False)
    consistency_config = resolve_consistency_config(
        test_data, os.environ.get("SMLA_BATCH_CONSISTENCY", "auto")
    )
    if consistency_config is not None:
        report = run_configured_consistency(
            test_data,
            consistency_config,
            lambda data: RunResult(*sparse_flash_mla_process.call_npu(data)),
            result_compare_method.check_result,
            lambda: torch_npu.npu.set_device(0),
        )
        summary = format_consistency_summary(report)
        print(summary, flush=True)
        fulfill_percent = consistency_fulfill_percent(report)
        final_result = "Passed" if report["pass"] else "Failed"
        utils.save_result(
            final_result, fulfill_percent, test_data["params"], result_path
        )
        if not report["relations"]:
            pytest.skip(f"batch consistency has no applicable relation: {report}")
        if not report["pass"]:
            pytest.fail(summary)
        return
    npu_result = None
    try:
        npu_result, softmax_lse = sparse_flash_mla_process.call_npu(test_data)
    except Exception as e:
        utils.save_result("Exception", 0, test_data["params"], result_path)
        pytest.fail(f"NPU执行异常: {e}")

    global_failed = False
    fulfill_percent = 0
    if npu_result is not None:
        result, fulfill_percent = result_compare_method.check_result(
            test_data["cpu_output"], npu_result
        )
        if result == "Failed":
            global_failed = True
    else:
        global_failed = True
        fulfill_percent = 0

    if test_data["params"].get("return_softmax_lse"):
        print("return_softmax_lse is true!!!")
        lse_result, lse_percent = result_compare_method.check_result(
            test_data["softmax_lse"], softmax_lse
        )
        if lse_result == "Failed":
            global_failed = True
        fulfill_percent = min(fulfill_percent, lse_percent)

    if global_failed:
        final_result = "Failed"
    else:
        final_result = "Passed"

    utils.save_result(final_result, fulfill_percent, test_data["params"], result_path)
    if final_result not in ("Passed", "passed", "Pass", "pass"):
        pytest.fail(
            f"用例结果校验失败: result={final_result}, fulfill_percent={fulfill_percent}"
        )
