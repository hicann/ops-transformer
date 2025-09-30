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


# coreNum, n, cols, k
case0_params = [40,160,96,1450,180,192,12,-1,0,0,0,256,1,40,5824,1,5824,5824,4864,1,4864,4864,6080,10,1024,40,5800,5800,1,5800,5800,1,5800,5800,40,5800,5800,1,5800,5800,1,5800,5800,1,96,96]
case1_params = [40,1,83,27,180,192,12,1,0,1,1,256,1,1,27,1,27,27,27,1,27,27,6080,0,1024,27,1,1,1,1,1,1,1,1,27,1,1,1,1,1,1,1,1,1,83,83]
case2_params = [40,2730,6144,8,185,193,8,-1,1,1,0,256,1,4,5472,1,5472,5472,5424,1,5440,5424,6080,0,1024,40,546,546,1,546,546,1,546,546,40,546,546,1,546,546,1,546,546,1,6144,6144]
case3_params = [40,3156,6144,8,116,124,8,-1,1,1,0,256,1,16,1600,1,1600,1600,1248,1,1248,1248,6080,4,1024,40,632,600,1,632,632,1,600,600,40,632,600,1,632,632,1,600,600,1,6144,6144]
case4_params = [40, 32, 674, 5205, 107, 428, 321, -1, 0, 1, 0, 1024, 1, 40, 4192, 1, 4192, 4192, 3072, 1, 3072, 3072, 6080, 10, 1024, 40, 4164, 4164, 1, 4164, 4164, 1, 4164, 4164, 40, 4164, 4164, 1, 4164, 4164, 1, 4164, 4164, 1, 674, 674]
case5_params = [40, 35, 2505, 8, 0, 620, 620, 1, 0, 0, 0, 1024, 1, 1, 280, 1, 280, 280, 280, 1, 280, 280, 6080, 0, 1024, 40, 7, 7, 1, 7, 7, 1, 7, 7, 40, 7, 7, 1, 7, 7, 1, 7, 7, 1, 2505, 2505]
case6_params = [40, 1, 7168, 8, 0, 256, 256, 1, 1, 1, 0, 256, 2, 1, 8, 1, 8, 8, 8, 1, 8, 8, 6112, 0, 1024, 8, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7168, 7168]


params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "case6": case6_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()