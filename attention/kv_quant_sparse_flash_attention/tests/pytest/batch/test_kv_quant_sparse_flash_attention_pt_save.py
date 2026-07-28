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

from pathlib import Path
import concurrent.futures
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import kv_quant_sparse_flash_attention_golden


PT_SAVE_PATH = "./pt_files/"
EXCEL_PATH = os.environ.get("EXCEL_PATH", "./excel/example.xlsx")
EXCEL_SHEET = os.environ.get("EXCEL_SHEET", "Sheet1")
RESULT_PATH = Path("result.xlsx")
ENABLED_PARAMS = utils.load_excel_test_cases(EXCEL_PATH, EXCEL_SHEET)
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS, pytest_paramset=False)
case_id = 0


def execute_qsfa(param_combination):
    global case_id
    params = utils.convert_param_combination_to_cs_format(param_combination)
    input_dict = kv_quant_sparse_flash_attention_golden.generate_input_tensors(params)
    cpu_result, _, _ = kv_quant_sparse_flash_attention_golden.compute_cpu(
        input_dict, params
    )
    test_data = {
        "Testcase_Name": params["case_name"],
        "params": params,
        "input": input_dict,
        "cpu_output": cpu_result,
    }
    kv_quant_sparse_flash_attention_golden._save_test_case(test_data, PT_SAVE_PATH)
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_kv_quant_sparse_flash_attention_pt_save(param_combination):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa, param_combination)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail("当前用例线程执行失败")
