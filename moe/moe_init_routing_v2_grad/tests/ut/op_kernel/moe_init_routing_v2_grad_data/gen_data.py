#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import numpy as np
import tensorflow as tf

DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32
}

def moe_init_routing_v2_grad(grad_expanded_x: np.ndarray,
                             expanded_row_idx: np.ndarray,
                             num_rows: int,
                             k: int,
                             hidden_size: int,
                             e: int,
                             c: int,
                             drop_pad_mode: int,
                             active_num: int) -> (np.ndarray):
    grad_x = np.zeros((num_rows, hidden_size), dtype = np.float32)
    for i in range(num_rows):
        for j in range(i * k, i * k + k, 1):
            expanded_x_idx = expanded_row_idx[j]

            if drop_pad_mode == 1:
                if expanded_x_idx == -1:
                    continue
            elif active_num > 0:
                if expanded_x_idx >= active_num:
                    continue
            grad_x[i] = np.add(grad_x[i], grad_expanded_x[expanded_x_idx].astype(np.float32))

    return grad_x

def gen_golden_data_simple(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, dtype):
    num_rows = int(num_rows)
    k = int(k)
    hidden_size = int(hidden_size)
    active_num = int(hidden_size)
    input_dtype = DTYPE.get(dtype, np.float32)
    output_dtype = input_dtype if (dtype != "bfloat16") else tf.bfloat16.as_numpy_dtype

    expand_n = active_num if (active_num > 0) else num_rows * k
    input_expanded_row_idx = np.random.choice(num_rows * k, num_rows * k, replace=False).astype(np.int32)
    if drop_pad_mode == 1:
        if dtype == "bfloat16":
            input_grad_expanded_x = tf.random.uniform([e, c, hidden_size], dtype=tf.bfloat16, maxval=1, minval=0)
            input_grad_expanded_x = input_grad_expanded_x.numpy()
        else:
            input_grad_expanded_x = np.random.uniform(0, 1, size=(e, c, hidden_size)).astype(input_dtype)
    else:
        if dtype == "bfloat16":
            input_grad_expanded_x = tf.random.uniform([expand_n, hidden_size], dtype=tf.bfloat16, maxval=1, minval=0)
            input_grad_expanded_x = input_grad_expanded_x.numpy()
        else:
            input_grad_expanded_x = np.random.uniform(0, 1, size=(expand_n, hidden_size)).astype(input_dtype)

    grad_x = moe_init_routing_v2_grad(input_grad_expanded_x, input_expanded_row_idx, num_rows, k, hidden_size, e, c, drop_pad_mode, active_num)
    grad_x = grad_x.astype(output_dtype)

    input_grad_expanded_x.tofile("./input_expanded_x.bin")
    input_expanded_row_idx.tofile("./input_expanded_row_idx.bin")

# python3 gen_data.py 8 2 5120 0 0 0 0 float32
if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])