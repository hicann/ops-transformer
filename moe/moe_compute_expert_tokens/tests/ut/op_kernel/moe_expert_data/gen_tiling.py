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
import numpy as np
import sys

case0_params = [
    48, 48, 48, 1, 102400, 10240, 381, 8, 0, 0, 0, 5, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1001, 48 * 32
]

case1_params = [
    48, 39, 0, 34, 102400, 20480, 77, 2, 1, 2, 2, 1, 1, 1, 1, 100, 3, 1, 3, 3, 1, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1002, 48 * 32]

case2_params = [
    40, 40, 39, 40, 196352, 409600, 97601, 2441, 8, 320, 201, 2402, 8, 320, 162, 8193, 205, 1, 205, 205, 198, 1,
    198, 198, 2560, 321, 8, 1, 320, 321, 9, 1024, 1, 1, 320, 1, 1, 1003, 48 * 32,
]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
}

def main():
    case = sys.argv[1]
    print("moe compute expert tokens tiling case: ", case)
    tiling = np.array(params_info.get(case), dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    tiling.tofile(tiling_file)


if __name__ == '__main__':
    main()
