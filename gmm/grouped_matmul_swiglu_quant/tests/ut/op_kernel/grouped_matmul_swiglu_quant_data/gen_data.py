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
import os
import numpy as np
import random

np.random.seed(0)


def gen_input_data(group_num, e, m, k, n):
    e = int(e)
    m = int(m)
    k = int(k)
    n = int(n)
    group_num = int(group_num)
    x = np.random.uniform(-5, 5, [m, k]).astype(np.int8)
    weight = np.random.uniform(-5, 5, [e, k, n]).astype(np.int8)
    xScale = np.random.uniform(
        -5,
        5,
        [
            m,
        ],
    ).astype(np.float32)
    weightScale = np.random.uniform(-5, 5, [e, n]).astype(np.float32)
    groupedList = np.linspace(1, m, group_num, dtype=np.int64)
    return x, weight, xScale, weightScale, y, groupedList


def gen_golden_data(group_num, e, m, k, n):
    x, weight, xScale, weightScale, y, groupedList = gen_input_data(
        group_num, e, m, k, n
    )
    x.tofile("x.bin")
    weight.tofile("weight.bin")
    weightScale.tofile("weightScale.bin")
    xScale.tofile("xScale.bin")
    y.tofile("y.bin")
    groupedList.tofile("groupedList.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
