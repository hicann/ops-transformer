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

import pytest
import torch

import result_compare_method
import utils
from batch import quant_block_sparse_attn_process
from batch.quant_block_sparse_attn_paramset_batch import load_cases_from_csv


TESTCASE_PATH = "bsa_testcase"
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0


def _testcase_files():
    cases = load_cases_from_csv()
    if not cases:
        print("error: no enabled cases found in CSV")
        return []
    files = []
    for case_name in cases:
        pt_filepath = os.path.join(TESTCASE_PATH, f"bsa_case_{case_name}.pt")
        if os.path.isfile(pt_filepath):
            files.append(pt_filepath)
        else:
            print(f"warning: pt file not found for case: {case_name}, skipped")
    if files:
        print(f"found {len(files)} testcase files (from {len(cases)} CSV cases)")
    return files


def bsa(testcase_file):
    test_data = torch.load(testcase_file, map_location="cpu")
    try:
        npu_result, cpu_result = quant_block_sparse_attn_process.test_quant_block_sparse_attn_process_ci(
            test_data, device_id=DEVICE_ID)
        result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    except pytest.skip.Exception:
        raise
    except Exception as error:
        print("NPU ERROR:", error)
        result = "NPU ERROR"
        fulfill_percent = 0

    utils.save_result(test_data["params"], result, fulfill_percent, RESULT_PATH)

    if result != "Pass":
        pytest.fail(f"case failed: {test_data['Testcase_Name']}")


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", _testcase_files())
def test_quant_block_sparse_attn(testcase_file):
    bsa(testcase_file)
