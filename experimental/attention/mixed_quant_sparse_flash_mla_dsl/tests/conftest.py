# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import os


def pytest_addoption(parser):
    group = parser.getgroup("mqsmla")
    group.addoption(
        "--excel",
        dest="mqsmla_excel",
        default=os.environ.get("MQSMLA_EXCEL"),
        help="Excel workbook; replaces test_case_paramset when specified",
    )
    group.addoption(
        "--sheet",
        dest="mqsmla_sheet",
        default=os.environ.get("MQSMLA_SHEET", "decode"),
        help="Excel sheet name (default: decode)",
    )
    group.addoption(
        "--data-mode",
        choices=("run", "save", "replay", "save-run"),
        default="run",
        help="Generate/run, save CPU data, replay .pt, or save then replay each case",
    )
    group.addoption(
        "--pt-dir",
        default=os.environ.get("MQSMLA_PT_DIR"),
        help="Directory for saved CPU inputs and golden (.pt)",
    )
    group.addoption(
        "--device-id", type=int, default=0, help="NPU device index (default: 0)"
    )
    group.addoption(
        "--metadata-backend",
        choices=("aicpu", "python"),
        default=os.environ.get("MQSMLA_METADATA_BACKEND", "aicpu"),
        help="MQSMLA metadata implementation (default: aicpu)",
    )
    group.addoption(
        "--verify-metadata-backends",
        action="store_true",
        default=os.environ.get("MQSMLA_VERIFY_METADATA_BACKENDS", "0") == "1",
        help="Check that Python and AICPU metadata partitions are identical",
    )
