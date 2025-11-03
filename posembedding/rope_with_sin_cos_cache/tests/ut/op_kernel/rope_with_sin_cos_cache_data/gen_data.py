#!/usr/bin/python3
# -*- coding:utf-8 -*-
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
import torch
import tensorflow as tf
import os

bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(0)

def gen_golden_data_simple(numTokens, numQHeads, numKHeads, headSize, rotaryDim, dtype):
    d_type_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": bf16
    }

    dtype = d_type_dict.get(dtype)

    data_position = np.random.uniform(1, int(numTokens), [int(numTokens)]).astype(np.int64)
    data_query = np.random.uniform(-2, 2, [int(numTokens), int(numQHeads) * int(headSize)]).astype(dtype)
    data_key = np.random.uniform(-2, 2, [int(numTokens), int(numKHeads) * int(headSize)]).astype(dtype)
    data_cosSinCache = np.random.uniform(-2, 2, [int(numTokens), int(rotaryDim)]).astype(dtype)

    data_position.tofile("./input_position.bin")
    data_query.tofile("./input_query.bin")
    data_key.tofile("./input_key.bin")
    data_cosSinCache.tofile("./input_cosSinCache.bin")

    data_queryOut = data_query
    data_keyOut = data_key

    data_queryOut.tofile("./output_queryOut.bin")
    data_keyOut.tofile("./output_keyOut.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6])
