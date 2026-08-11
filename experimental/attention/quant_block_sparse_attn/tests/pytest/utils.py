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

import numpy as np
import pandas as pd
import pytest


REQUIRED_EXCEL_COLUMNS = ["layout_q", "layout_kv", "B", "S1", "S2", "N1", "N2", "D"]
LEGACY_RESULT_COLUMN_RENAMES = {
    "kv_layout": "layout_kv",
    "q_block_size": "sparse_q_block_size",
    "kv_block_size": "sparse_kv_block_size",
    "sparse_indices_layout": "layout_sparse_indices",
}


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    if sheetname is None:
        sheetname = "Sheet1"

    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})
        missing_cols = [col for col in REQUIRED_EXCEL_COLUMNS if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)
        return [row.to_dict() for _, row in df.iterrows()]
    except Exception as error:
        pytest.skip(f"Failed to read Excel file: {error}", allow_module_level=True)
        return []


def _row_from_dict(params, result, fulfill_percent):
    keys = [
        "Testcase_Name",
        "layout_q",
        "layout_kv",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "sparse_q_block_size",
        "sparse_kv_block_size",
        "sparse_mode",
        "softmax_scale",
        "layout_sparse_indices",
        "layout_out",
        "quant_mode",
        "mask_mode",
        "return_softmax_lse",
        "sparse_pattern",
        "block_table_pattern",
        "block_num",
        "q_datarange",
        "k_datarange",
        "v_datarange",
        "pa_block_padding_bytes",
        "p_scale_value",
        "seed",
        "FLOPS",
        "MFU*时间",
    ]
    row_data = {"case_name": params.get("Testcase_Name")}
    row_data.update({key: params.get(key) for key in keys if key != "Testcase_Name"})
    row_data["result"] = result
    row_data["fulfill_percent"] = fulfill_percent
    return row_data


def _row_from_legacy_tuple(params, result, fulfill_percent):
    row_data = {
        "case_name": params[0],
        "layout_q": params[1],
        "layout_kv": params[2],
        "B": params[6],
        "S1": params[7],
        "N1": params[9],
        "N2": params[10],
        "D": params[11],
        "softmax_scale": params[19],
    }
    row_data["result"] = result
    row_data["fulfill_percent"] = fulfill_percent
    return row_data


def _append_result_row(existing_df, row_data):
    current_columns = list(row_data.keys())
    if existing_df is None:
        return pd.DataFrame([row_data], columns=current_columns)

    df = existing_df.rename(columns=LEGACY_RESULT_COLUMN_RENAMES)
    if set(df.columns) != set(current_columns):
        missing_columns = [column for column in current_columns if column not in df.columns]
        extra_columns = [column for column in df.columns if column not in current_columns]
        print("warning: result columns do not match current case fields, merging columns")
        print(f"Missing columns: {missing_columns}")
        print(f"Extra columns: {extra_columns}")
        for column in missing_columns:
            df[column] = None
        for column in extra_columns:
            row_data[column] = None
        current_columns.extend(extra_columns)
    df = df.reindex(columns=current_columns)
    return pd.concat([df, pd.DataFrame([row_data], columns=current_columns)], ignore_index=True)


def _save_result_csv_fallback(row_data, result_path, error):
    csv_path = result_path.with_suffix(".csv")
    existing_df = pd.read_csv(csv_path) if csv_path.exists() else None
    df = _append_result_row(existing_df, row_data)
    df.to_csv(csv_path, index=False)
    print(f"warning: Excel result requires optional dependency ({error}); wrote {csv_path}")
    return True


def save_result(params, result, fulfill_percent, result_path):
    if isinstance(params, dict):
        row_data = _row_from_dict(params, result, fulfill_percent)
    else:
        row_data = _row_from_legacy_tuple(params, result, fulfill_percent)

    try:
        existing_df = pd.read_excel(result_path) if result_path.exists() else None
        df = _append_result_row(existing_df, row_data)
        df.to_excel(result_path, index=False)
    except ImportError as error:
        return _save_result_csv_fallback(row_data, result_path, error)
    return True
