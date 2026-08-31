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
from functools import reduce
import math
import torch


INVALID_VALUE = -1


def get_torch_dtype(type_str):
    type_dict = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "hifloat8": torch.uint8,
    }
    if type_str == "float8_e4m3fn":
        return torch.float8_e4m3fn
    return type_dict[type_str]


def get_torch_compute_dtype(type_str):
    compute_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.float32,
        "bfloat16": torch.float32,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "hifloat8": torch.float32,
        "float8_e4m3fn": torch.float32,
    }
    return compute_map.get(type_str, torch.float32)


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
    "sinks": 7,
}


def gen_tensor_data(params, key):
    index = SHAPE_KEY_TO_INDEX.get(key, 0)
    shape_input = params["shape_input"][key]
    gen_dtype = params["dtype_input"][key]
    compute_dtype = get_torch_compute_dtype(gen_dtype)

    is_fix_value = False
    input_i_data = None
    for param_key in params.keys():
        if f"tensor_data_{index}" in param_key and "required_" not in param_key:
            input_i_data = torch.tensor(params[param_key], dtype=compute_dtype).reshape(
                shape_input
            )
            is_fix_value = True

    if not is_fix_value:
        if len(params["range_input"][key]) == 1:
            input_i_data = gen_boundary_tensor_data(
                shape_input,
                params["range_input"][key],
                compute_dtype,
                params["dtype_input"][key],
            )
        else:
            range_input = params["range_input"][key]
            input_i_data = gen_nonbound_tensor_data(
                params, shape_input, range_input, compute_dtype, index
            )

            if params["dtype_input"][key] == "hifloat8":
                input_i_data = trans_np_float_tensor_to_hifuint8(
                    in_tensor=input_i_data, round_mode="hybrid", over_mode=True
                )

    if input_i_data is None:
        return None

    if not torch.is_tensor(input_i_data):
        input_i_data = torch.tensor(input_i_data, dtype=compute_dtype)

    if gen_dtype == "hifloat8":
        return input_i_data

    final_dtype = get_torch_dtype(gen_dtype)
    if input_i_data.dtype != final_dtype:
        input_i_data = input_i_data.to(final_dtype)
    return input_i_data


def trans_np_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    multi_shape = in_tensor.numel()
    flat = in_tensor.reshape(-1).float()
    out_tensor = torch.zeros(multi_shape, dtype=torch.uint8)
    for i in range(multi_shape):
        out_tensor[i] = cvt_float32_to_hifuint8(flat[i].item(), round_mode, over_mode)
    return out_tensor.reshape(shape_tensor)


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
    t14_mask = 16383
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192
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
    if math.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return 239
            else:
                return 238
        else:
            if over_mode:
                return 111
            else:
                return 110
    if math.isnan(x):
        if over_mode:
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
        input_i_data = torch.tensor([], dtype=dtype_input)
    elif input_range[0] == "nan":
        input_i_data = torch.full(shape_input, float("nan"), dtype=dtype_input)
    elif input_range[0] == "inf":
        input_i_data = torch.full(shape_input, float("inf"), dtype=dtype_input)
    elif input_range[0] == "-inf":
        input_i_data = torch.full(shape_input, float("-inf"), dtype=dtype_input)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        input_i_data = torch.full(shape_input, input_range[0], dtype=dtype_input)
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
            if dtype in [torch.float32, torch.float64, torch.float16]:
                min_value = torch.finfo(dtype).min
                max_value = torch.finfo(dtype).max
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                min_value = torch.iinfo(dtype).min
                max_value = torch.iinfo(dtype).max
            else:
                min_value = False
                max_value = True
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            input_data = (
                torch.empty(data_shape, dtype=torch.float32)
                .uniform_(float(min_value), float(max_value))
                .to(dtype)
            )
            if data_shape == []:
                return gen_uniform_data(data_shape, -1, 1, dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if shape_len == 0:
                return gen_uniform_data(data_shape, 0, 4, dtype, hash_seed)
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            num_change = torch.randint(0, min(shape_len, 4), (1,)).item()
            flat = input_data.reshape(-1)
            if num_change > 0:
                if hash_seed > 0:
                    torch.manual_seed(hash_seed + 1)
                inf_index = torch.randint(0, shape_len, (num_change,))
                flat[inf_index] = float(min_value)
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 2)
            num_change = torch.randint(0, min(shape_len, 4), (1,)).item()
            if num_change > 0:
                if hash_seed > 0:
                    torch.manual_seed(hash_seed + 3)
                inf_index = torch.randint(0, shape_len, (num_change,))
                flat[inf_index] = float(max_value)
            return input_data
        elif min_value == "nan":
            return torch.full(data_shape, float("nan"), dtype=dtype)
        elif min_value == "inf":
            return torch.full(data_shape, float("inf"), dtype=dtype)
        elif min_value == "-inf":
            return torch.full(data_shape, float("-inf"), dtype=dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return torch.tensor([], dtype=dtype)
        else:
            return gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed)
    except MemoryError:
        print("[ERROR] MemoryError.")
        return INVALID_VALUE


def gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if hash_seed > 0:
        torch.manual_seed(hash_seed)
    data = (
        torch.empty(data_shape, dtype=torch.float32)
        .uniform_(float(min_value), float(max_value))
        .to(dtype)
    )
    return data
