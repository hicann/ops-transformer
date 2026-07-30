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

import utils
import os
import pandas as pd
from pathlib import Path
import pytest
import sparse_flash_mla_golden
import traceback

excel_path = os.getenv("SMLA_EXCEL_PATH", "./excel/example.xlsx")
excel_sheet = os.getenv("SMLA_EXCEL_SHEET", "CSA")
ENABLED_PARAMS_FROM_FILE = utils.load_excel_test_cases(excel_path, excel_sheet)
save_path = os.getenv("SMLA_PT_SAVE_PATH", "./data")

param_combinations = utils.generate_param_combinations(
    ENABLED_PARAMS_FROM_FILE, is_save_pt=True
)

case_id = 0
failed_cases = []
failed_record_path = Path("pt_save_failed.xlsx")


def record_failed_case(param_combinations, error_msg):
    global case_id
    row = {
        "case_id": case_id,
        "testcase_name": param_combinations.get("testcase_name"),
        "template_mode": param_combinations.get("template_mode"),
        "layout_q": param_combinations.get("layout_q"),
        "layout_kv": param_combinations.get("layout_kv"),
        "B": param_combinations.get("B"),
        "S1": param_combinations.get("S1"),
        "S2": param_combinations.get("S2"),
        "N1": param_combinations.get("N1"),
        "K": param_combinations.get("K"),
        "K1": param_combinations.get("K1"),
        "cmp_ratio": param_combinations.get("cmp_ratio"),
        "ori_mask_mode": param_combinations.get("ori_mask_mode"),
        "cmp_mask_mode": param_combinations.get("cmp_mask_mode"),
        "error_msg": str(error_msg),
    }
    failed_cases.append(row)
    df = pd.DataFrame(failed_cases)
    df.to_excel(failed_record_path, index=False)


def generate_and_save(param_combinations):
    global case_id
    try:
        test_data = utils.generate_case_with_default_param(param_combinations)
        print("data parsed.", test_data)
        print("strat to generate data")
        input_data = sparse_flash_mla_golden.gen_data(test_data)
        print("strat to save data")
        sparse_flash_mla_golden.save_test_case(input_data, save_path)
    except Exception as e:
        record_failed_case(param_combinations, e)
        pytest.fail(
            f"[FAILED CASE RECORDED] case_id={case_id}\n{traceback.format_exc()}"
        )
    finally:
        case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_sparse_flash_mla(param_combinations):
    generate_and_save(param_combinations)
