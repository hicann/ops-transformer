#!/usr/bin/python3
# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import glob
import os
import tensorflow as tf

bf16 = tf.bfloat16.as_numpy_dtype

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file, output_file, d_type):
    if d_type == "float16":
        precision = 1 / 1000
    else:
        precision = 1 / 10000
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bf16,
        "bf16fp32": bf16,
    }
    scale_d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bf16,
        "bf16fp32": np.float32,
    }
    dtype = d_type_dict.get(d_type)
    scale_dtype = scale_d_type_dict.get(d_type)

    golden = np.fromfile(golden_file, dtype)
    output = np.fromfile(output_file, dtype)

    diff_res = np.isclose(output, golden, precision, 0, True)
    diff_idx = np.where(diff_res != True)[0]
    diff_ratio = len(diff_idx) / golden.size
    if len(diff_idx) == 0 or diff_ratio < precision:
        print("PASSED!")

    else:
        print("FAILED!")
        print(f"Diff ratio: {len(diff_idx) / golden.size}")
        for idx in diff_idx[:5]:
            print(f"index: {idx}, output: {output[idx]}, golden: {golden[idx]}")

    return


def process(d_type):
    golden_file = os.path.join(curr_dir, "out_golden.bin")
    output_file = os.path.join(curr_dir, "output.bin")
    result = compare_data(golden_file, output_file, d_type)


if __name__ == "__main__":
    process(sys.argv[1])
