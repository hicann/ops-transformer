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

import numpy as np
import sys

#           num_rows:   topk:   hidden_size:
# case0:    8           2       5120
# case1:    4096        40      8
# case2:    1           48      1024
# case3:    20          20      44321

# coreNum, n, cols, k
case0_params = [40, 8, 0, 0, 5120, 2, 0, 8, 1, 1, 1, 5120, 5120, 1, 1, 0, 20480, 1, 1, 1, 1, 1]
case1_params = [40, 4096, 0, 0, 8, 40, 0, 40, 103, 79, 1, 16, 8, 16, 1, 4, 64, 103, 1, 103, 1, 79]
case2_params = [40, 10, 0, 0, 512, 64, 40, 10, 1, 1, 1, 512, 512, 32, 1, 5, 2048, 1, 1, 1, 1, 1]
case3_params = [40, 80, 10, 8, 512, 8, 40, 40, 2, 2, 1, 512, 512, 4, 1, 2, 2048, 2, 1, 2, 1, 2]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()