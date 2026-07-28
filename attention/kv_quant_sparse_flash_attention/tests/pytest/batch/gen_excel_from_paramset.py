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
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


PARAMSET_FILE = os.environ.get(
    "PARAMSET_FILE", "kv_quant_sparse_flash_attention_paramset"
)
EXCEL_OUTPUT_PATH = os.environ.get("EXCEL_OUTPUT_PATH", "./excel/example.xlsx")
EXCEL_SHEET = os.environ.get("EXCEL_SHEET", "Sheet1")

ENABLED_PARAMS = utils.load_paramset(PARAMSET_FILE)
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS, pytest_paramset=True)

EXCEL_COLUMNS = [
    "Testcase_Prefix",
    "layout_query",
    "layout_kv",
    "q_type",
    "kv_type",
    "B",
    "T1",
    "T2",
    "S1",
    "S2",
    "N1",
    "N2",
    "D",
    "K",
    "scale_value",
    "key_quant_mode",
    "value_quant_mode",
    "sparse_block_size",
    "tile_size",
    "rope_head_dim",
    "sparse_mode",
    "attention_mode",
    "quant_scale_repo_mode",
    "block_size",
    "block_num",
    "actual_seq_q",
    "actual_seq_kv",
    "range_query",
    "range_key",
    "range_query_rope",
    "range_key_rope",
    "range_dequant_scale",
]


def convert_value_for_excel(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def gen_excel_from_paramset():
    rows = []
    for param_combo in PARAM_COMBINATION_SET:
        row = {}
        for col in EXCEL_COLUMNS:
            value = param_combo.get(col)
            row[col] = convert_value_for_excel(value)
        rows.append(row)

    df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)

    output_dir = os.path.dirname(EXCEL_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_excel(EXCEL_OUTPUT_PATH, index=False, sheet_name=EXCEL_SHEET)
    print(f"生成 Excel 文件: {EXCEL_OUTPUT_PATH}")
    print(f"Sheet 名称: {EXCEL_SHEET}")
    print(f"用例数量: {len(rows)}")


if __name__ == "__main__":
    gen_excel_from_paramset()
