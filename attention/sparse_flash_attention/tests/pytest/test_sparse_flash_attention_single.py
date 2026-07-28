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
import utils
import sparse_flash_attention_golden

PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")

PARAMSET_FILE = os.environ.get("PARAMSET_FILE", "sparse_flash_attention_paramset")
ENABLED_PARAMS = utils.load_paramset(PARAMSET_FILE)
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_sfa(param_combination):
    global case_id
    params = utils.convert_param_combination_to_cs_format(param_combination)
    input_dict = sparse_flash_attention_golden.generate_input_tensors(params)
    cpu_result, _, _ = sparse_flash_attention_golden.compute_cpu(input_dict, params)
    test_data = {
        "Testcase_Name": params["case_name"],
        "params": params,
        "input": input_dict,
        "cpu_output": cpu_result,
    }
    if SAVE_PT:
        sparse_flash_attention_golden._save_test_case(test_data, PT_SAVE_PATH)
    result, compare_results = utils.sfa_run_npu(
        test_data, testcase_name=None, device_id=DEVICE_ID, result_path=RESULT_PATH
    )
    case_id += 1
    return result, compare_results, test_data


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_sparse_flash_attention_single(param_combination):
    test_data = None
    compare_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_sfa, param_combination)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result, compare_results, test_data = future.result()
                if result == "Failed":
                    case_name = test_data.get("Testcase_Name", "unknown")
                    return_softmax_lse = test_data.get("input", {}).get(
                        "return_softmax_lse", False
                    )
                    compare_items = ["attn_out"]
                    if return_softmax_lse:
                        compare_items.extend(["softmax_max", "softmax_sum"])
                    detail_parts = []
                    for item in compare_items:
                        item_result = compare_results.get(item, {})
                        item_status = item_result.get("result", "Unknown")
                        item_percent = item_result.get("fulfill_percent", 0.0)
                        detail_parts.append(
                            f"{item}: {item_status}({item_percent:.4f}%)"
                        )
                    detail_msg = ", ".join(detail_parts)
                    pytest.fail(
                        f"用例名: {case_name}, 对比结果: {detail_msg}", pytrace=False
                    )
            except Exception as e:
                params = test_data.get("params") if test_data else None
                case_name = (
                    test_data.get("Testcase_Name", "unknown")
                    if test_data
                    else "unknown"
                )
                if params:
                    utils.save_result(params, "Failed", "", RESULT_PATH, None, None)
                pytest.fail(f"用例名: {case_name}, 当前用例线程执行失败")
