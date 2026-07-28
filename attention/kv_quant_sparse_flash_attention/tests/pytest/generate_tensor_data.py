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
from ml_dtypes import float8_e4m3fn


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
        "hifloat8": np.float32,
    }
    if type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    else:
        return type_dict[type_str]


SHAPE_KEY_TO_INDEX = {
    "query": 0,
    "key": 1,
    "value": 2,
    "sparse_indices": 3,
    "block_table": 4,
    "query_cache": 5,
    "key_cache": 6,
    "value_cache": 7,
    "query_rope": 8,
    "key_rope": 9,
    "dequant_scale": 10,
    "v_dequant_scale": 11,
}


def _np_to_torch(np_array, gen_dtype):
    """将 numpy array 转为 torch tensor，处理 bf16/float8 等特殊类型。"""
    torch_dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "int8": torch.int8,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "hifloat8": torch.uint8,
    }
    if gen_dtype in ("bf16", "bfloat16"):
        return torch.from_numpy(np.array(np_array).astype(np.float32)).to(
            torch.bfloat16
        )
    if gen_dtype == "float8_e4m3fn":
        return torch.from_numpy(np.array(np_array).astype(np.float32)).to(
            torch.float8_e4m3fn
        )
    torch_dtype = torch_dtype_map.get(gen_dtype)
    if torch_dtype is not None:
        return torch.from_numpy(np.array(np_array)).to(torch_dtype)
    # 其他类型（如 fp4、hifloat8 等），保留为 numpy 并包装
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

            if params["dtype_input"][key] == "hifloat8":
                input_i_data = trans_np_float_tensor_to_hifuint8(
                    in_tensor=input_i_data, round_mode="hybrid", over_mode=True
                )

    return _np_to_torch(input_i_data, gen_dtype)


def trans_np_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    if multi_shape == 1.0:  # shape为空时，输出multi_shape可能为浮点数 1.0
        multi_shape = int(multi_shape)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.astype(np.uint8)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor


def _get_hif8_fraction_bits_number(exponent):
    if exponent < -22:
        return -1, 3, 0
    if -22 <= exponent < -15:
        return 0, 3, 0
    if exponent == 0:
        return 1, 0, 3
    if abs(exponent) == 1:
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        return 12, 4, 1
    if exponent > 15:
        return 12, 4, -1


def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383  # b11111111111111
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192  # 10 0000 0000 0000
        t14_values = fraction32_int & t14_mask
        hif8_value = 0

    else:
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if np.isinf(x) or x_abs >= over_value:
        # 2^15 = 32768
        if sign:
            if over_mode:
                # b11101111 = 239
                return 239
            else:
                # b11101110 = 238
                return 238
        else:
            if over_mode:
                # b01101111 = 111
                return 111
            else:
                # b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            # b10000000
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(x_abs * pow(2, 23) * pow(2, -exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = (
        _get_hif8_fraction_bits_number(exponent)
    )
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp32_ssr_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    else:
        print("unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = (
            _get_hif8_fraction_bits_number(exponent)
        )
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < -23:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = (
            sign_int_value
            + dot_int_value
            + sig_exp
            + exponent_int_value
            + hif8_frac_value
        )
    return hif8_int_value


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
            if data_shape == []:  # shape=[] reduce处理会异常
                return gen_uniform_data(data_shape, -1, 1, dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if shape_len == 0:  # high不能为0
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
