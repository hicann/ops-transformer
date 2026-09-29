# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import os
from pathlib import Path

import pytest

import result_compare_method
from batch import quant_lightning_indexer_v2_pt_loadprocess
from qliv2_test_utils import (
    QliV2CaseSelector,
    QliV2ResultWriter,
    ensure_comparison_passed,
)


TEST_INPUT_PATH = os.environ.get("QLIV2_TESTCASE_DIR", "pt_path").strip()
SINGLE_CASE_PATH = os.environ.get("QLIV2_TESTCASE_PATH", "").strip()
RESULT_PATH = os.environ.get("QLIV2_RESULT_PATH", "").strip()
TESTCASE_FILES = []
if SINGLE_CASE_PATH:
    TESTCASE_FILES = QliV2CaseSelector.resolve(
        TEST_INPUT_PATH, explicit_files=SINGLE_CASE_PATH
    )
elif os.path.isdir(TEST_INPUT_PATH):
    TESTCASE_FILES = QliV2CaseSelector.resolve(
        TEST_INPUT_PATH,
        explicit_files=os.environ.get("QLIV2_PT_FILE_LIST", ""),
        case_names=os.environ.get("QLIV2_CASE_NAMES", ""),
        case_indexes=os.environ.get("QLIV2_CASE_INDEXES", ""),
    )


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_qliv2(testcase_file):
    print(f"执行用例：{testcase_file}")
    cpu, npu, scores, cpu_values, npu_values, offset, params = (
        quant_lightning_indexer_v2_pt_loadprocess.test_qliv2_process(
            testcase_file, device_id=0
        )
    )
    result, percent = result_compare_method.check_result(
        cpu, npu, scores, offset, params, cpu_values, npu_values
    )
    value_result, value_percent = result_compare_method.check_result_return_value(
        cpu_values, npu_values, params, cpu, npu, scores, offset
    )
    print(f"result: {result}")
    print(f"fulfill_percent: {percent}")
    print(f"result_return_value: {value_result}")
    print(f"fulfill_percent_return_value: {value_percent}")
    case_name = Path(testcase_file).stem
    if RESULT_PATH:
        QliV2ResultWriter.append(
            RESULT_PATH,
            QliV2ResultWriter.row(
                case_name, params, result, percent, value_result, value_percent
            ),
        )
    ensure_comparison_passed(case_name, result, percent, value_result, value_percent)
