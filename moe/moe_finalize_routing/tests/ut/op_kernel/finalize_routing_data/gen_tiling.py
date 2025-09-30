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

case0_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2004] #db float
case1_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2005] #db float16
case2_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2003] #db bfloat16
case3_params = [40, 40, 0, 10, 1024, 8936, 7008, 1928, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 2006] #cuthk2 float
case4_params = [40, 40, 0, 10, 1024, 8936, 7008, 1928, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 2007] #cuthk2 float16
case5_params = [40, 10, 0, 10, 10, 6400, 5152, 1248, 1, 0, 
                0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2008] #cuthk2 bfloat16
case6_params = [40, 40, 0, 10, 1024, 8936, 3200, 2536, 2, 0, 
                0, 0, 4, 26, 78, 1, 1, 10, 30, 1, 1, 2009] #cuthk4 float
case7_params = [40, 40, 0, 10, 1024, 8936, 3200, 2536, 2, 0, 
                0, 0, 4, 26, 78, 1, 1, 10, 30, 1, 1, 20010] #cuthk4 float16
case8_params = [40, 10, 0, 10, 10, 6400, 3152, 96, 2, 0,
                0, 0, 4, 1, 3, 1, 1, 1, 3, 1, 1, 20011] #cuthk4 bfloat16
case9_params = [40, 40, 0, 10, 1024, 8936, 3200, 2536, 2, 0, 
                0, 0, 3, 26, 78, 1, 1, 10, 30, 1, 1, 20012] #cuth float
case10_params = [40, 40, 0, 10, 1024, 8936, 3200, 2536, 2, 0, 
                0, 0, 3, 26, 78, 1, 1, 10, 30, 1, 1, 20013] #cuth float16
case11_params = [40, 10, 0, 10, 10, 6400, 5152, 1248, 1, 0, 
                0, 0, 3, 1, 2, 1, 1, 1, 2, 1, 1, 20014] #cuth bfloat16
case12_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20000] #CUTK float
case13_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20001] #CUTK float16
case14_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20002] #CUTK bfloat16

case15_params = [40, 39, 0, 471, 933, 1756, 1756, 1756, 0, 0, 0, 0, 4, 24,
                4, 7, 3, 21, 3, 7, 7, 20015] #AllBias float32

case16_params = [40, 38, 0, 702, 340, 2617, 2617, 2617, 0, 0, 0, 0, 5, 9,
                1, 9, 9, 7, 1, 7, 7, 20016] #AllBias float16

case17_params = [40, 35, 0, 592, 105, 1328, 1328, 1328, 0, 0, 0, 0, 6, 3,
                1, 3, 3, 3, 1, 3, 3, 20017] #AllBias bfloat16

case18_params = [40, 12, 0, 16, 12, 5120, 2560, 2560, 1, 0, 0, 0, 4, 1,
                1, 1, 1, 1, 1, 1, 1, 20018] #optimized float32

case0_params = [2, 2, 3, 12, 7, 2, 6, 2, 4, 2, 6, 2, 4, 2, 0]
params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "case6": case6_params,
    "case7": case7_params,
    "case8": case8_params,
    "case9": case9_params,
    "case10": case10_params,
    "case11": case11_params,
    "case12": case12_params,
    "case13": case13_params,
    "case14": case14_params,
    "case15": case15_params,
    "case16": case16_params,
    "case17": case17_params,
    "case18": case18_params
}


def main():
    case = sys.argv[1]
    print("moe finalize tiling case: ", case)
    tiling = np.array(params_info.get(case), dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    tiling.tofile(tiling_file)


if __name__ == '__main__':
    main()
