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

def gen_golden_data_simple(N, H, k, E, dtype):
    N = int(N)
    H = int(H)
    k = int(k)
    E = int(E)

    input_x = np.random.uniform(-2, 2, [N, H]).astype(dtype)
    input_expertIdx = np.random.randint(0, E, size=(N, k)).astype("int32")
    scale = np.random.uniform(-2, 2, [N]).astype(np.float32)

    input_x.tofile("./input_x.bin")
    input_expertIdx.tofile("./input_expertIdx.bin")
    scale.tofile("./scale.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])