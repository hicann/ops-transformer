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

import operator
import random
from functools import reduce
import numpy as np
import os
import json
import math
import struct
import torch
import tensorflow as tf
import copy


INVALID_VALUE = -1


def get_np_dtype(type_str):
    type_dict = {
        "fp32": np.float32,
        "float32": np.float32,
        "fp16": np.float16,
        "float16": np.float16,
        "bf16": tf.bfloat16.as_numpy_dtype,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    return type_dict[type_str]


SHAPE_KEY_TO_INDEX = {
    "query": 0,
    "key": 1,
    "value": 2,
    "sparse_indices": 3,
    "block_table": 4,
    "query_rope": 5,
    "key_rope": 6,
    "sinks": 7,
}


def _np_to_torch(np_array, gen_dtype):
    """将 numpy array 转为 torch tensor，处理 bf16 等特殊类型。"""
    torch_dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "int8": torch.int8,
        "int32": torch.int32,
        "uint8": torch.uint8,
    }
    if gen_dtype in ("bf16", "bfloat16"):
        return torch.from_numpy(np.array(np_array).astype(np.float32)).to(
            torch.bfloat16
        )
    torch_dtype = torch_dtype_map.get(gen_dtype)
    if torch_dtype is not None:
        return torch.from_numpy(np.array(np_array)).to(torch_dtype)
    return torch.from_numpy(np.array(np_array))


def gen_tensor_data(params, key):
    input_i_data = None
    index = SHAPE_KEY_TO_INDEX.get(key, 0)
    shape_input = params["shape_input"][key]
    gen_dtype = params["dtype_input"][key]
    dtype_input = get_np_dtype(gen_dtype)

    is_fix_value = False
    for param_key in params.keys():
        if f"tensor_data_{index}" in param_key and "required_" not in param_key:
            input_i_data = (
                np.array(params[param_key]).reshape(shape_input).astype(gen_dtype)
            )
            is_fix_value = True

    if not is_fix_value:
        dtype_input = get_np_dtype(gen_dtype)
        tensor_duplicate_rat = None
        real_shape = None
        diff_data_size = None

        if len(params["range_input"][key]) == 1:
            input_i_data = gen_boundary_tensor_data(
                shape_input,
                params["range_input"][key],
                dtype_input,
                params["dtype_input"][key],
            )
        else:
            range_input = params["range_input"][key]
            input_i_data = gen_nonbound_tensor_data(
                params, shape_input, range_input, dtype_input, index
            )

    return _np_to_torch(input_i_data, gen_dtype)


def gen_boundary_tensor_data(shape_input, input_range, dtype_input, dtype_raw="fp32"):
    input_i_data = None
    if input_range[0] == "null":
        input_i_data = np.array([], dtype=dtype_input)
    elif input_range[0] == "nan":
        try:
            input_i_data = np.full(shape_input, np.NaN, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, np.nan, dtype=dtype_input)
    elif input_range[0] == "inf":
        try:
            input_i_data = np.full(shape_input, np.Inf, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, np.inf, dtype=dtype_input)
    elif input_range[0] == "-inf":
        try:
            input_i_data = np.full(shape_input, -np.Inf, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, -np.inf, dtype=dtype_input)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        input_i_data = np.full(shape_input, input_range[0], dtype=dtype_input)
    return input_i_data


def gen_nonbound_tensor_data(params, data_shape, range_input, dtype, tensor_index):
    fuzz_value_type = ""
    try:
        if isinstance(range_input, dict):
            fuzz_value_type = "normal"
        elif isinstance(range_input, list):
            fuzz_value_type = "uniform"
        else:
            return INVALID_VALUE
        hash_seed = -1
        min_value, max_value = range_input
        if min_value == "-inf" and max_value == "inf":
            if dtype in [np.float32, np.float64, np.float16]:
                min_value = np.finfo(dtype).min
                max_value = np.finfo(dtype).max
            elif dtype in [np.int8, np.int16, np.int32, np.uint8]:
                min_value = np.iinfo(dtype).min
                max_value = np.iinfo(dtype).max
            else:
                min_value = False
                max_value = True
            if hash_seed > 0:
                np.random.seed(hash_seed)
            if dtype in [np.float64]:
                input_data = (
                    np.random.uniform(low=-1, high=1, size=data_shape).astype(
                        np.float64
                    )
                    * max_value
                )
            else:
                input_data = np.random.uniform(
                    low=min_value, high=max_value, size=data_shape
                ).astype(dtype)
            if data_shape == []:
                return gen_uniform_data(data_shape, -1, 1, dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if shape_len == 0:
                return gen_uniform_data(data_shape, 0, 4, dtype, hash_seed)
            if hash_seed > 0:
                np.random.seed(hash_seed)
            num_change = np.random.randint(low=0, high=min(shape_len, 4))
            if hash_seed > 0:
                np.random.seed(hash_seed + 1)
            inf_index = np.unravel_index(
                np.random.choice(shape_len, num_change), data_shape
            )
            input_data[inf_index] = min_value
            if hash_seed > 0:
                np.random.seed(hash_seed + 2)
            num_change = np.random.randint(low=0, high=min(shape_len, 4))
            if hash_seed > 0:
                np.random.seed(hash_seed + 3)
            inf_index = np.unravel_index(
                np.random.choice(shape_len, num_change), data_shape
            )
            input_data[inf_index] = max_value
            return input_data
        elif min_value == "nan":
            try:
                return np.full(data_shape, np.NaN, dtype=dtype)
            except Exception:
                return np.full(data_shape, np.nan, dtype=dtype)
        elif min_value == "inf":
            try:
                return np.full(data_shape, np.Inf, dtype=dtype)
            except Exception:
                return np.full(data_shape, np.inf, dtype=dtype)
        elif min_value == "-inf":
            try:
                return np.full(data_shape, -np.Inf, dtype=dtype)
            except Exception:
                return np.full(data_shape, -np.inf, dtype=dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return np.array([], dtype=dtype)
        else:
            return gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed)
    except MemoryError:
        print("[ERROR] MemoryError.")
        return INVALID_VALUE


def gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if hash_seed > 0:
        np.random.seed(hash_seed)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    if dtype == np.complex64:
        real = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
            np.float32
        )
        real = torch.tensor(real)
        if hash_seed > 0:
            np.random.seed(hash_seed + 1)
        imag = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
            np.float32
        )
        imag = torch.tensor(imag)
        input_complex64 = torch.complex(real, imag)
        data = input_complex64.detach().numpy().astype(np.complex64)
    data = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )
    return data
