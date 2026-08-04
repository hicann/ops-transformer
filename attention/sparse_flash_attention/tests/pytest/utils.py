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
import itertools
import math
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import check_valid_param
import result_compare_method
import sparse_flash_attention_golden
from batch import sparse_flash_attention_process


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}


def _normalize_numeric_value(value):
    # Excel 混合空值/整数列时，pandas 容易把整列抬升成 float，这里统一回收成 int。
    if isinstance(value, list):
        return [_normalize_numeric_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numeric_value(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_excel_cell_value(value):
    # 将 Excel 单元格内容转换成 pytest 框架需要的 Python 类型。
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in STR_MAP_DICT:
            return _normalize_numeric_value(STR_MAP_DICT[stripped])
        if stripped in ("", "None", "none", "NULL", "null", "NaN", "nan"):
            return None
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("(") and stripped.endswith(")")
        ):
            try:
                return _normalize_numeric_value(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                return value
    return _normalize_numeric_value(value)


def load_paramset(paramset_file):
    module = __import__(paramset_file)
    return module.ENABLED_PARAMS


def generate_actual_seq(S, B, layout=None, T=None):
    """
    根据S和B自动生成actual_seq。

    参数:
        S: 序列长度上限（S1或S2）
        B: batch大小
        layout: layout类型，TND场景下采用累加和
        T: 当不为None时，将actual_seq最后一个数改为T

    返回:
        actual_seq: 生成的实际序列长度列表
    """
    if layout == "TND":
        if T is not None:
            seq_lengths = []
            remaining = T
            for i in range(B - 1):
                max_len = min(S, remaining)
                min_len = max(0, remaining - (B - i - 1) * S)
                length = random.randint(min_len, max_len)
                seq_lengths.append(length)
                remaining -= length
            seq_lengths.append(remaining)

            random.shuffle(seq_lengths)
            actual_seq = []
            cumsum = 0
            for length in seq_lengths:
                cumsum += length
                actual_seq.append(cumsum)
        else:
            seq_lengths = [random.randint(0, S) for _ in range(B)]
            actual_seq = []
            cumsum = 0
            for length in seq_lengths:
                cumsum += length
                actual_seq.append(cumsum)
    else:
        seq_lengths = [random.randint(0, S) for _ in range(B)]
        actual_seq = seq_lengths

    return actual_seq


def load_excel_test_cases(excel_file_path, sheet_name):
    # batch 模式从 Excel 读取原始参数行，并校验字段完整性。
    if sheet_name is None:
        sheet_name = "Sheet1"
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        dataframe = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        dataframe = dataframe.replace({np.nan: None, pd.NA: None})
    except Exception as error:
        pytest.skip(f"Failed to read Excel file: {error}", allow_module_level=True)

    required_columns = [
        "Testcase_Prefix",
        "layout_query",
        "layout_kv",
        "q_type",
        "kv_type",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "scale_value",
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "return_softmax_lse",
        "block_size",
        "block_num",
        "actual_seq_q",
        "actual_seq_kv",
    ]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        pytest.skip(
            f"Missing required columns in Excel: {missing_columns}",
            allow_module_level=True,
        )
    return [row.to_dict() for _, row in dataframe.iterrows()]


def save_result(
    params,
    result,
    fulfill_percent,
    result_path,
    softmax_max_percent=None,
    softmax_sum_percent=None,
):
    """保存测试结果，列与 example.xlsx 一致，结果文件可直接用于批量生成 pt。"""
    row_data = {
        "Testcase_Prefix": params.get("Testcase_Prefix", "sparseFlashAttn"),
        "layout_query": params.get("layout_query"),
        "layout_kv": params.get("layout_kv"),
        "q_type": str(params.get("q_type")),
        "kv_type": str(params.get("kv_type"))
        if params.get("kv_type") is not None
        else None,
        "B": params.get("B"),
        "T1": params.get("T1"),
        "T2": params.get("T2"),
        "S1": params.get("S1"),
        "S2": params.get("S2"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K"),
        "scale_value": params.get("scalevalue"),
        "sparse_block_size": params.get("sparse_blocksize"),
        "rope_head_dim": params.get("rope_head_dim"),
        "sparse_mode": params.get("sparsemode"),
        "attention_mode": params.get("attention_mode"),
        "block_size": params.get("block_size"),
        "block_num": params.get("block_num"),
        "actual_seq_q": str(params.get("actual_seq_q"))
        if params.get("actual_seq_q") is not None
        else None,
        "actual_seq_kv": str(params.get("actual_seq_kv"))
        if params.get("actual_seq_kv") is not None
        else None,
        "range_query": str(params.get("range_query"))
        if params.get("range_query") is not None
        else None,
        "range_key": str(params.get("range_key"))
        if params.get("range_key") is not None
        else None,
        "range_query_rope": str(params.get("range_query_rope"))
        if params.get("range_query_rope") is not None
        else None,
        "range_key_rope": str(params.get("range_key_rope"))
        if params.get("range_key_rope") is not None
        else None,
        "range_sinks": str(params.get("range_sinks"))
        if params.get("range_sinks") is not None
        else None,
        "use_sinks": params.get("use_sinks", False),
        "return_softmax_lse": params.get("return_softmax_lse", False),
        "result": result,
        "fulfill_percent": fulfill_percent,
        "softmax_max_percent": softmax_max_percent,
        "softmax_sum_percent": softmax_sum_percent,
    }

    if result_path.exists():
        dataframe = pd.read_excel(result_path)
        dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    else:
        dataframe = pd.DataFrame([row_data])
    dataframe.to_excel(result_path, index=False)


def combin_params(enabled_params, pytest_paramset=True):
    param_combination_set = []
    base_param_names = [
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
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "return_softmax_lse",
        "use_sinks",
        "block_size",
        "block_num",
        "actual_seq_q",
        "actual_seq_kv",
    ]
    range_param_names = [
        "range_query",
        "range_key",
        "range_query_rope",
        "range_key_rope",
        "range_sinks",
    ]

    for params in enabled_params:
        current_params = {}
        for key, value in params.items():
            if key in base_param_names or key in range_param_names:
                current_params[key] = (
                    value if pytest_paramset else [_parse_excel_cell_value(value)]
                )

        param_names = base_param_names + range_param_names
        param_values = [current_params.get(name, [None]) for name in param_names]

        for combo in itertools.product(*param_values):
            param_combination_set.append(dict(zip(param_names, combo)))
    return param_combination_set


def convert_param_combination_to_cs_format(param_combination):
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    T1 = param_combination.get("T1")
    B = param_combination["B"]
    S1 = param_combination["S1"]
    T2 = param_combination.get("T2")
    S2 = param_combination["S2"]
    N1 = param_combination["N1"]
    N2 = param_combination["N2"]
    D = param_combination["D"]
    K = param_combination["K"]
    q_type = param_combination["q_type"]
    kv_type = param_combination["kv_type"]
    scale_value = param_combination["scale_value"]
    sparse_block_size = param_combination["sparse_block_size"]
    rope_head_dim = param_combination["rope_head_dim"]
    sparse_mode = param_combination["sparse_mode"]
    attention_mode = param_combination["attention_mode"]
    return_softmax_lse = param_combination.get("return_softmax_lse", False)
    use_sinks = bool(param_combination.get("use_sinks", False))
    block_size = param_combination.get("block_size") or 256
    block_num = param_combination.get("block_num")

    # 参数校验：TND layout 下 T 不能超过 B*S
    if layout_query == "TND" and T1 is not None:
        if T1 > B * S1:
            raise ValueError(f"Invalid parameter: T1={T1} exceeds B*S1={B * S1}")

    if layout_kv == "TND" and T2 is not None:
        if T2 > B * S2:
            raise ValueError(f"Invalid parameter: T2={T2} exceeds B*S2={B * S2}")

    if param_combination.get("actual_seq_q") is None:
        actual_seq_q = generate_actual_seq(S1, B, layout_query, T1)
    else:
        actual_seq_q = param_combination.get("actual_seq_q")

    if layout_query == "TND":
        T1 = actual_seq_q[-1]

    if param_combination.get("actual_seq_kv") is None:
        actual_seq_kv = generate_actual_seq(S2, B, layout_kv, T2)
    else:
        actual_seq_kv = param_combination.get("actual_seq_kv")

    if layout_kv == "TND":
        T2 = actual_seq_kv[-1]

    sparse_blockcount = int(K / sparse_block_size)
    testcase_prefix = param_combination.get("Testcase_Prefix") or "sparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or 1
    return_softmax_lse = param_combination.get("return_softmax_lse", False)

    if layout_kv == "PA_BSND":
        block_num_per_batch = math.ceil(S2 / block_size)
    if layout_kv == "PA_BSND" and block_num is None:
        block_num = 0
        for length in actual_seq_kv:
            block_num = block_num + math.ceil(length / block_size)

    q_dtype_str = "bf16" if q_type == torch.bfloat16 else "fp16"
    testcase_name = f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_dtype_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"
    kv_dtype_str = "bf16" if kv_type == torch.bfloat16 else "fp16"
    if layout_kv == "PA_BSND":
        if layout_query == "BSND":
            shape_input = {
                "query": [B, S1, N1, D],
                "key": [B, S2, N2, D],
                "value": [B, S2, N2, D],
                "sparse_indices": [B, S1, N2, sparse_blockcount],
                "block_table": [B, block_num_per_batch],
                "query_cache": [B, S1, N1, D + rope_head_dim],
                "key_cache": [block_num, block_size, N2, D + rope_head_dim],
                "value_cache": [block_num, block_size, N2, D + rope_head_dim],
                "query_rope": [B, S1, N1, rope_head_dim],
                "key_rope": [B, S2, N2, rope_head_dim],
                "sinks": [N1],
            }
            shape_output = {
                "attn_out": [B, S1, N1, D],
                "softmax_max": [B, N2, S1, int(N1 / N2)],
                "softmax_sum": [B, N2, S1, int(N1 / N2)],
            }
        elif layout_query == "TND":
            shape_input = {
                "query": [T1, N1, D],
                "key": [B, S2, N2, D],
                "value": [B, S2, N2, D],
                "sparse_indices": [T1, N2, sparse_blockcount],
                "block_table": [B, block_num_per_batch],
                "query_cache": [T1, N1, D + rope_head_dim],
                "key_cache": [block_num, block_size, N2, D + rope_head_dim],
                "value_cache": [block_num, block_size, N2, D + rope_head_dim],
                "query_rope": [T1, N1, rope_head_dim],
                "key_rope": [B, S2, N2, rope_head_dim],
                "sinks": [N1],
            }
            shape_output = {
                "attn_out": [T1, N1, D],
                "softmax_max": [N2, T1, int(N1 / N2)],
                "softmax_sum": [N2, T1, int(N1 / N2)],
            }
        else:
            print("Unsupported layout_query: ", layout_query)
    elif layout_kv == "TND":
        shape_input = {
            "query": [T1, N1, D],
            "key": [T2, N2, D],
            "value": [T2, N2, D],
            "sparse_indices": [T1, N2, sparse_blockcount],
            "block_table": [B],
            "query_cache": [T1, N1, D + rope_head_dim],
            "key_cache": [T2, N2, D + rope_head_dim],
            "value_cache": [T2, N2, D + rope_head_dim],
            "query_rope": [T1, N1, rope_head_dim],
            "key_rope": [T2, N2, rope_head_dim],
            "sinks": [N1],
        }
        shape_output = {
            "attn_out": [T1, N1, D],
            "softmax_max": [N2, T1, int(N1 / N2)],
            "softmax_sum": [N2, T1, int(N1 / N2)],
        }
    elif layout_kv == "BSND":
        shape_input = {
            "query": [B, S1, N1, D],
            "key": [B, S2, N2, D],
            "value": [B, S2, N2, D],
            "sparse_indices": [B, S1, N2, sparse_blockcount],
            "block_table": [B],
            "query_cache": [B, S1, N1, D + rope_head_dim],
            "key_cache": [B, S2, N2, D + rope_head_dim],
            "value_cache": [B, S2, N2, D + rope_head_dim],
            "query_rope": [B, S1, N1, rope_head_dim],
            "key_rope": [B, S2, N2, rope_head_dim],
            "sinks": [N1],
        }
        shape_output = {
            "attn_out": [B, S1, N1, D],
            "softmax_max": [B, N2, S1, int(N1 / N2)],
            "softmax_sum": [B, N2, S1, int(N1 / N2)],
        }
    else:
        print("Unsupported layout_kv: ", layout_kv)
    dtype_input = {
        "query": q_dtype_str,
        "key": kv_dtype_str,
        "value": kv_dtype_str,
        "sparse_indices": "int32",
        "block_table": "int32",
        "query_cache": q_dtype_str,
        "key_cache": kv_dtype_str,
        "value_cache": kv_dtype_str,
        "query_rope": q_dtype_str,
        "key_rope": q_dtype_str,
        "sinks": "fp32",
    }

    default_range_input = {
        "query": [-10.0, 100.0],
        "key": [5.0, 100.0],
        "sparse_indices": [-10, 10],
        "block_table": [0, 1],
        "query_rope": [-10.0, 10.0],
        "key_rope": [-10.0, 10.0],
        "sinks": [-0.2, 0.2],
    }
    range_input = {}
    for key in ["query", "key", "query_rope", "key_rope", "sinks"]:
        range_key = f"range_{key}"
        if range_key in param_combination and param_combination[range_key] is not None:
            range_input[key] = param_combination[range_key]
        else:
            range_input[key] = default_range_input[key]
    range_input["sparse_indices"] = default_range_input["sparse_indices"]
    range_input["block_table"] = default_range_input["block_table"]

    params = {
        "case_name": testcase_name,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_seq_q,
        "actualseqlengthskv": actual_seq_kv,
        "scalevalue": scale_value,
        "sparsemode": sparse_mode,
        "sparse_blocksize": sparse_block_size,
        "shape_input": shape_input,
        "dtype_input": dtype_input,
        "range_input": range_input,
        "dtype_output": [q_dtype_str],
        "shape_output": shape_output,
        "rope_head_dim": rope_head_dim,
        "attention_mode": attention_mode,
        "block_size": block_size,
        "return_softmax_lse": return_softmax_lse,
        "use_sinks": use_sinks,
        # 原始参数，用于结果保存（与 example.xlsx 列对齐）
        "Testcase_Prefix": testcase_prefix,
        "q_type": q_type,
        "kv_type": kv_type,
        "B": B,
        "T1": T1,
        "T2": T2,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K": K,
        "block_num": block_num,
        "actual_seq_q": actual_seq_q,
        "actual_seq_kv": actual_seq_kv,
        # range 参数，保存实际使用的值
        "range_query": range_input["query"],
        "range_key": range_input["key"],
        "range_query_rope": range_input["query_rope"],
        "range_key_rope": range_input["key_rope"],
        "range_sinks": range_input["sinks"],
    }

    return params


def sfa_run_npu(test_data, testcase_name=None, device_id=0, result_path=None):
    params = test_data.get("params")
    if testcase_name:
        params["Testcase_Prefix"] = testcase_name
    cpu_result = test_data["cpu_output"]
    compare_results = {}
    softmax_max_percent = None
    softmax_sum_percent = None
    try:
        npu_result = sparse_flash_attention_process.call_npu(
            test_data["input"], params, device_id=device_id
        )
        print("[compare] comparing attn_out ...")
        result_attn_out, fulfill_percent_attn = result_compare_method.check_result(
            cpu_result[0], npu_result[0]
        )
        compare_results["attn_out"] = {
            "result": result_attn_out,
            "fulfill_percent": fulfill_percent_attn,
        }
        overall_result = result_attn_out
        overall_fulfill_percent = fulfill_percent_attn
        if test_data["input"]["return_softmax_lse"]:
            print("[compare] comparing softmax_max ...")
            result_softmax_max, fulfill_percent_max = (
                result_compare_method.check_result(cpu_result[1], npu_result[1])
            )
            print("[compare] comparing softmax_sum ...")
            result_softmax_sum, fulfill_percent_sum = (
                result_compare_method.check_result(cpu_result[2], npu_result[2])
            )
            compare_results["softmax_max"] = {
                "result": result_softmax_max,
                "fulfill_percent": fulfill_percent_max,
            }
            compare_results["softmax_sum"] = {
                "result": result_softmax_sum,
                "fulfill_percent": fulfill_percent_sum,
            }
            softmax_max_percent = fulfill_percent_max
            softmax_sum_percent = fulfill_percent_sum
            overall_result = (
                "Pass"
                if (
                    result_attn_out == "Pass"
                    and result_softmax_max == "Pass"
                    and result_softmax_sum == "Pass"
                )
                else "Failed"
            )
    except Exception:
        if result_path:
            save_result(
                params,
                "Failed",
                "",
                result_path,
                softmax_max_percent,
                softmax_sum_percent,
            )
        raise
    if result_path:
        save_result(
            params,
            overall_result,
            overall_fulfill_percent,
            result_path,
            softmax_max_percent,
            softmax_sum_percent,
        )
    return overall_result, compare_results
