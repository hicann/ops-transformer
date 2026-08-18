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
import pytest
from pathlib import Path
import os
import result_compare_method
from batch import quant_sparse_flash_mla_process
import utils
import logging

testcase_path = os.environ.get("QSMLA_PT_DIR", "qsmla_testcase")
is_run_graph = os.environ.get("RUN_GRAPH", "0") == "1"
result_path = os.environ.get("QSMLA_RESULT_SAVE_PATH", "result.xlsx")
device_id = 0

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(filename)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

testcase_files = []
if os.path.isdir(testcase_path):
    pt_files = [f for f in os.listdir(testcase_path) if f.endswith(".pt")]
    if not pt_files:
        logging.warning(f"错误: 目录中没有找到.pt文件: {testcase_path}")
    else:
        logging.info(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(testcase_path, pt_file)
            testcase_files.append(filepath)
else:
    logging.warning(f"错误: 输出目录不存在: {testcase_path}")


def qsmla(testcase_files):
    test_data = torch.load(testcase_files, map_location="cpu", weights_only=False)
    try:
        if is_run_graph:
            npu_result, cpu_quant_result, cpu_lse, npu_lse = (
                quant_sparse_flash_mla_process.test_qsmla_quant_process_graph(
                    test_data, device_id=device_id
                )
            )
        else:
            npu_result, cpu_quant_result, cpu_lse, npu_lse = (
                quant_sparse_flash_mla_process.test_qsmla_quant_process_ci(
                    test_data, device_id=device_id
                )
            )

        # 分别校验主输出与LSE，分开保存结果
        main_res, main_pct = result_compare_method.check_result(
            cpu_quant_result, npu_result
        )
        lse_res, lse_pct = None, None
        fail_info = []
        min_fulfill = main_pct

        if test_data["params"].get("return_softmax_lse"):
            print("return_softmax_lse is true!!!")
            lse_res, lse_pct = result_compare_method.check_result(cpu_lse, npu_lse)

        # 主输出失败记录
        if main_res != "Pass":
            fail_info.append(f"MAIN_FAILED:{main_res}")
        # LSE失败记录
        if lse_res is not None and lse_res != "Pass":
            fail_info.append(f"LSE_FAILED:{lse_res}")
            min_fulfill = min(min_fulfill, lse_pct)

        # 整合最终结果
        if fail_info:
            result = "; ".join(fail_info)
            fulfill_percent = min_fulfill
        else:
            result = "PASS"
            fulfill_percent = min_fulfill

    except Exception as e:
        logging.exception(e)
        result = "NPU ERROR"
        fulfill_percent = 0

    utils.save_result(test_data["params"], result, fulfill_percent, Path(result_path))

    if result != "PASS":
        pytest.fail(
            f"用例执行失败:{test_data['Testcase_Name']} 精度:{fulfill_percent:.2f}%"
        )


testcase_ids = [os.path.splitext(os.path.basename(f))[0] for f in testcase_files]


@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", testcase_files, ids=testcase_ids)
def test_quant_sparse_flash_mla(testcase_files):
    qsmla(testcase_files)
