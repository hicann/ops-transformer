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
from pathlib import Path
import numpy as np
import random
import tensorflow as tf

bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(0)


def gen_input_data(dtype, group_num, m, k, n):
    d_type_dict = {"fp16": np.float16, "bf16": bf16}
    m = int(m)
    k = int(k)
    n = int(n)
    group_num = int(group_num)

    print(1)
    dtype = d_type_dict.get(dtype)
    x = np.random.uniform(-5, 5, [k, m]).astype(dtype)
    weight = np.random.uniform(-5, 5, [k, n]).astype(dtype)
    y = np.random.uniform(-5, 5, [m * group_num, n]).astype(np.float32)
    groupedList = np.linspace(1, k, group_num, dtype=np.int64)
    return x, weight, y, groupedList


def gen_golden_data(dtype, group_num, m, k, n):
    x, weight, y, groupedList = gen_input_data(dtype, group_num, m, k, n)
    x.tofile("x.bin")
    weight.tofile("weight.bin")
    y.tofile("y.bin")
    groupedList.tofile("groupedList.bin")


if __name__ == "__main__":
    for file_name in ("x.bin", "weight.bin", "y.bin", "groupedList.bin"):
        generated_file = Path(file_name)
        if generated_file.is_file():
            generated_file.unlink()
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
