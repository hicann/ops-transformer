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

import ast
import os

EXCEL_COLUMNS = [
    "case_name", "enable",
    "B", "N1", "N2", "S1", "S2",
    "cu_seqlens_q_value", "cu_seqlens_kv_value",
    "seqused_q_value", "seqused_kv_value",
    "D",
    "sparse_q_block_size", "sparse_kv_block_size",
    "sparse_mode", "sparse_pattern",
    "block_table_pattern",
    "block_num",
    "max_block_per_batch",
    "q_datarange", "k_datarange", "v_datarange",
    "layout_q", "layout_kv", "layout_sparse_indices", "layout_out",  "output_dtype",
    "quant_mode", "mask_mode",
    "p_scale_value", "softmax_scale",
    "return_softmax_lse", "seed",
]

PA_BLOCK_PADDING_BYTES = 0

EXCEL_FILENAME = "cases.csv,cases_stc.csv,cases_generalized.csv"

_INT_COLS = {"B", "N1", "N2", "S1", "S2", "D",
             "sparse_q_block_size", "sparse_kv_block_size",
             "quant_mode", "mask_mode", "seed",
             "block_num", "max_block_per_batch"}
_FLOAT_COLS = {"p_scale_value", "softmax_scale"}
_BOOL_COLS = {"return_softmax_lse"}
_INT_LIST_COLS = {
    "cu_seqlens_q_value", "cu_seqlens_kv_value",
    "seqused_q_value", "seqused_kv_value",
}


def _get_excel_paths(csv_path=None):
    if csv_path is None:
        csv_path = EXCEL_FILENAME
    if isinstance(csv_path, (list, tuple)):
        csv_names = csv_path
    else:
        csv_names = str(csv_path).split(",")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_paths = []
    for csv_name in csv_names:
        csv_name = str(csv_name).strip()
        if not csv_name:
            continue
        if os.path.isabs(csv_name):
            csv_paths.append(csv_name)
        else:
            csv_paths.append(os.path.join(base_dir, csv_name))
    return csv_paths


def _get_excel_path():
    csv_paths = _get_excel_paths(EXCEL_FILENAME)
    return csv_paths[0] if csv_paths else ""


def _get_dtype_map():
    import torch as _torch
    return {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() == "TRUE"
    return bool(val)


def _parse_int(val):
    if val is None:
        return None
    return int(val)


def _parse_float(val):
    if val is None:
        return None
    if isinstance(val, str) and val.strip().upper() == "NONE":
        return None
    return float(val)


def _parse_int_list(val):
    if val is None:
        return None
    try:
        values = ast.literal_eval(val) if isinstance(val, str) else val
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"expected an integer list, got {val!r}") from error
    if not isinstance(values, list):
        raise ValueError(f"expected an integer list, got {val!r}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ValueError(f"expected an integer list, got {val!r}")
    return list(values)


def _load_cases_from_one_csv(csv_path):
    import csv
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    header = [c.strip() for c in rows[0]]
    dtype_map = _get_dtype_map()
    import torch as _torch
    cases = {}
    for row in rows[1:]:
        record = dict(zip(header, row))
        case_name = record.get("case_name")
        if case_name is None:
            continue
        case_name = str(case_name).strip()
        if not case_name:
            continue

        if not _parse_bool(record.get("enable", False)):
            continue

        params = {}
        params["Testcase_Name"] = [case_name]
        for col in EXCEL_COLUMNS:
            if col in ("case_name", "enable"):
                continue
            val = record.get(col)
            if val == "":
                val = None
            if col == "sparse_mode" and val is None:
                val = "random"
            if col in _INT_COLS:
                val = _parse_int(val)
            elif col in _FLOAT_COLS:
                val = _parse_float(val)
            elif col in _BOOL_COLS:
                val = _parse_bool(val)
            elif col in _INT_LIST_COLS:
                val = _parse_int_list(val)
            elif col == "output_dtype":
                val = dtype_map.get(str(val).strip(), _torch.bfloat16) if val is not None else _torch.bfloat16
            params[col] = [val]

        params["pa_block_padding_bytes"] = [PA_BLOCK_PADDING_BYTES]
        params["S1EQS2"] = [params["S1"][0] == params["S2"][0]]
        cases[case_name] = params
    return cases


def load_cases_from_csv(csv_path=None):
    cases = {}
    for one_csv_path in _get_excel_paths(csv_path):
        cases.update(_load_cases_from_one_csv(one_csv_path))
    return cases


TEST_PARAMS = load_cases_from_csv()
ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
