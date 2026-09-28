#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str): 工作表名称。

    返回:
        list[tuple]: 测试用例元组列表。若失败或跳过，则返回空列表。
    """
    if sheetname is None:
        sheetname = "Sheet1"

    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        required_columns = [
            "layout",
            "dtype",
            "seqlen_q",
            "seqlen_kv",
            "n1",
            "n2",
            "head_dim",
            "rope_head_dim",
            "sparse_block_size",
            "sparse_block_count",
            "block_size",
            "page_attention",
            "return_softmax_lse",
            "return_float_output",
            "sparse_mode",
            "case_name",
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(
                f"Missing required columns in Excel: {missing_cols}",
                allow_module_level=True,
            )

        test_cases = []
        for _, row in df.iterrows():
            test_cases.append(row.to_dict())
        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None


def save_result(
    result, fulfill_percent, params, result_path="./result/sfa_result.xlsx"
):
    result_path = Path(result_path)
    row_data = {
        "case_name": params[15],
        "layout": params[0],
        "dtype": str(params[1]),
        "seqlen_q": params[2],
        "seqlen_kv": params[3],
        "n1": params[4],
        "n2": params[5],
        "head_dim": params[6],
        "rope_head_dim": params[7],
        "sparse_block_size": params[8],
        "sparse_block_count": params[9],
        "block_size": params[10],
        "page_attention": params[11],
        "return_softmax_lse": params[12],
        "return_float_output": params[13],
        "sparse_mode": params[14],
        "result": result,
        "fulfill_percent": fulfill_percent,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        df = pd.read_excel(result_path)
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            return False
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    df.to_excel(result_path, index=False)
    return True
