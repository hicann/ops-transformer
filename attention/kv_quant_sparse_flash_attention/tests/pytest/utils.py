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
import kv_quant_sparse_flash_attention_golden
import result_compare_method
import traceback
import tensorflow as tf
import json


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}


def _normalize_numeric_value(value):
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
        T: 当不为None时，将actual_seq最后一个数改为T（可传入T1或T2）

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


def save_result(params, result, fulfill_percent, result_path):
    """保存测试结果，列与 example.xlsx 一致，结果文件可直接用于批量生成 pt。"""
    row_data = {
        "Testcase_Prefix": params.get("Testcase_Prefix", "kvQuantSparseFlashAttn"),
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
        "key_quant_mode": params.get("key_quant_mode"),
        "value_quant_mode": params.get("value_quant_mode"),
        "sparse_block_size": params.get("sparse_blocksize"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "sparse_mode": params.get("sparsemode"),
        "attention_mode": params.get("attention_mode"),
        "quant_scale_repo_mode": params.get("quant_scale_repo_mode"),
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
        "range_dequant_scale": str(params.get("range_dequant_scale"))
        if params.get("range_dequant_scale") is not None
        else None,
        "result": result,
        "fulfill_percent": fulfill_percent,
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
    ]
    range_param_names = [
        "range_query",
        "range_key",
        "range_query_rope",
        "range_key_rope",
        "range_dequant_scale",
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

        print("param_values", param_values)
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
    key_quant_mode = param_combination["key_quant_mode"]
    value_quant_mode = param_combination["value_quant_mode"]
    sparse_block_size = param_combination["sparse_block_size"]
    tile_size = param_combination["tile_size"]
    rope_head_dim = param_combination["rope_head_dim"]
    sparse_mode = param_combination["sparse_mode"]
    attention_mode = param_combination["attention_mode"]
    quant_scale_repo_mode = param_combination["quant_scale_repo_mode"]
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
    testcase_prefix = (
        param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    )
    testcase_number = param_combination.get("Testcase_Number") or 1

    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"

    if layout_kv == "PA_BSND":
        block_num_per_batch = math.ceil(S2 / block_size)
    if layout_kv == "PA_BSND" and block_num is None:
        block_num = 0
        for length in actual_seq_kv:
            block_num = block_num + math.ceil(length / block_size)

    if q_type == torch.bfloat16:
        q_dtype_str = "bf16"
    elif q_type == torch.float16:
        q_dtype_str = "fp16"
    else:
        q_dtype_str = "fp32"

    if kv_type is None or str(kv_type) == "float8_e4m3fn":
        kv_dtype_str = "float8_e4m3fn"
    else:
        kv_dtype_str = str(kv_type)
    if layout_kv == "PA_BSND":
        if layout_query == "BSND":
            shape_input = {
                "query": [B, S1, N1, D],
                "key": [B, S2, N2, D],
                "value": [B, S2, N2, D],
                "sparse_indices": [B, S1, N2, sparse_blockcount],
                "block_table": [B, block_num_per_batch],
                "query_cache": [B, S1, N1, D + rope_head_dim],
                "key_cache": [
                    block_num,
                    block_size,
                    N2,
                    D + rope_head_dim * 2 + D // tile_size * 4,
                ],
                "value_cache": [
                    block_num,
                    block_size,
                    N2,
                    D + rope_head_dim * 2 + D // tile_size * 4,
                ],
                "query_rope": [B, S1, N1, rope_head_dim],
                "key_rope": [B, S2, N2, rope_head_dim],
                "dequant_scale": [B, S2, N2, D // tile_size],
                "v_dequant_scale": [B, S2, N2, D // tile_size],
            }
        elif layout_query == "TND":
            shape_input = {
                "query": [T1, N1, D],
                "key": [B, S2, N2, D],
                "value": [B, S2, N2, D],
                "sparse_indices": [T1, N2, sparse_blockcount],
                "block_table": [B, block_num_per_batch],
                "query_cache": [T1, N1, D + rope_head_dim],
                "key_cache": [
                    block_num,
                    block_size,
                    N2,
                    D + rope_head_dim * 2 + D // tile_size * 4,
                ],
                "value_cache": [
                    block_num,
                    block_size,
                    N2,
                    D + rope_head_dim * 2 + D // tile_size * 4,
                ],
                "query_rope": [T1, N1, rope_head_dim],
                "key_rope": [B, S2, N2, rope_head_dim],
                "dequant_scale": [B, S2, N2, D // tile_size],
                "v_dequant_scale": [B, S2, N2, D // tile_size],
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
            "key_cache": [T2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            "value_cache": [T2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            "query_rope": [T1, N1, rope_head_dim],
            "key_rope": [T2, N2, rope_head_dim],
            "dequant_scale": [T2, N2, D // tile_size],
            "v_dequant_scale": [T2, N2, D // tile_size],
        }
    elif layout_kv == "BSND":
        shape_input = {
            "query": [B, S1, N1, D],
            "key": [B, S2, N2, D],
            "value": [B, S2, N2, D],
            "sparse_indices": [B, S1, N2, sparse_blockcount],
            "block_table": [B],
            "query_cache": [B, S1, N1, D + rope_head_dim],
            "key_cache": [B, S2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            "value_cache": [B, S2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            "query_rope": [B, S1, N1, rope_head_dim],
            "key_rope": [B, S2, N2, rope_head_dim],
            "dequant_scale": [B, S2, N2, D // tile_size],
            "v_dequant_scale": [B, S2, N2, D // tile_size],
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
        "dequant_scale": "fp32",
        "v_dequant_scale": "fp32",
    }

    default_range_input = {
        "query": [2, 10],
        "key": [-100, 100.0],
        "sparse_indices": [-10, 10],
        "block_table": [0, 1],
        "query_rope": [-1, 1],
        "key_rope": [-10.0, -2],
        "dequant_scale": [0, 1],
        "v_dequant_scale": [0, 1],
    }
    range_input = {}
    for key in ["query", "key", "query_rope", "key_rope", "dequant_scale"]:
        range_key = f"range_{key}"
        if range_key in param_combination and param_combination[range_key] is not None:
            range_input[key] = param_combination[range_key]
        else:
            range_input[key] = default_range_input[key]
    range_input["sparse_indices"] = default_range_input["sparse_indices"]
    range_input["block_table"] = default_range_input["block_table"]
    range_input["v_dequant_scale"] = range_input["dequant_scale"]

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
        "shape_output": [[B, S1, N1, D]],
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "key_quant_mode": key_quant_mode,
        "value_quant_mode": value_quant_mode,
        "attention_mode": attention_mode,
        "quant_scale_repo_mode": quant_scale_repo_mode,
        "block_size": block_size,
        "k_antiquant_mode": key_quant_mode,
        "v_antiquant_mode": value_quant_mode,
        "antiquant_scale": 1,
        "antiquant_offset": 0,
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
        "range_dequant_scale": range_input["dequant_scale"],
    }

    return params


def get_np_dtype(type_str):
    type_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
        "bf16": tf.bfloat16.as_numpy_dtype,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        "float32": np.float32,
        "float16": np.float16,
        "hifloat8": np.uint8,
    }
    if type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    else:
        return type_dict[type_str]


def convert_tensor_data(data_pool, data_path, type_str, params=None):
    KNOW_NP_DTYPES = [
        "fp32",
        "fp16",
        "int32",
        "int8",
        "uint8",
        "bf16",
        "bfloat16",
        "float32",
        "float16",
    ]

    np_type = data_pool.dtype
    dump_np_data = None
    if len(np_type) > 1:
        dump_np_data = data_pool
    elif type_str.lower() in KNOW_NP_DTYPES:
        np_dtype = get_np_dtype(type_str.lower())
        dump_np_data = np.array(data_pool).astype(np_dtype)
    elif type_str.lower() in ["float8_e4m3fn", "hifloat8"]:
        dump_np_data = np.array(data_pool)
    else:
        dump_np_data = np.array(data_pool)
    if dump_np_data is not None:
        dump_shape = list(dump_np_data.shape)
        file_name = os.path.basename(data_path)
        if "cpu_output" in file_name and isinstance(params, type({})):
            if "shape_output_tmp" not in params.keys():
                params["shape_output_tmp"] = []
            params["shape_output_tmp"].append(dump_shape)
            shape_info = {"shape_output": params["shape_output_tmp"]}
            modify_cs_info(shape_info, params)
    return dump_np_data


def get_str_dtype(type_str):
    type_dict = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.int32: "int32",
        torch.uint8: "uint8",
        torch.bfloat16: "bf16",
    }
    return type_dict[type_str]


def get_torch_dtype(type_str):
    type_dict = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "hifloat8": None,
    }
    return type_dict.get(type_str)


def get_tensor_dtype(tensor, input_dtype):
    if input_dtype:
        tensor_dtype = input_dtype
    else:
        if torch.is_tensor(tensor):
            tensor_dtype = get_str_dtype(tensor.dtype)
        else:
            tensor_dtype = str(tensor.dtype)
    return tensor_dtype


def modify_alcnn_input_file(
    name: str, tensors, params: dict, data_range=None, data_dtype=None
):
    out_tensor = tensors
    if tensors is not None:
        data_range = data_range or [-10.0, 10.0]
        tensor_dtype = get_tensor_dtype(tensors, data_dtype)
        if torch.is_tensor(tensors):
            if tensor_dtype == "bf16":
                out_tensor = np.array(
                    tensors.float()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(tf.bfloat16.as_numpy_dtype)
                )
            else:
                out_tensor = tensors.detach().cpu().numpy()
        else:
            out_tensor = convert_tensor_data(tensors, name, tensor_dtype, params)

        params["shape_input"][name] = list(tensors.shape)
        params["dtype_input"][name] = tensor_dtype
        params["range_input"][name] = data_range

    return out_tensor


def qsfa_run_npu(test_data, testcase_name=None, device_id=0, result_path=None):
    from batch import kv_quant_sparse_flash_attention_process

    params = test_data.get("params")
    if testcase_name:
        params["Testcase_Prefix"] = testcase_name
    cpu_result = test_data["cpu_output"]
    try:
        npu_result = kv_quant_sparse_flash_attention_process.call_npu(
            test_data["input"], params
        )
        result, fulfill_percent = result_compare_method.check_result(
            cpu_result, npu_result
        )
    except Exception:
        if result_path:
            save_result(params, "Failed", "", result_path)
        raise
    if result_path:
        save_result(params, result, fulfill_percent, result_path)
    return result, fulfill_percent
