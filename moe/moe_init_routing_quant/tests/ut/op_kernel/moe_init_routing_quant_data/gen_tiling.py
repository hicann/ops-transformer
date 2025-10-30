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

#           num_rows:   topk:   hidden_size:    active_rows:
# case0:    8           2       5120            5120
# case1:    4096        40      8               8
# case2:    1           48      1024            1
# case3:    20          20      44321           20

# coreNum, n, cols, k
case0_params = [1, 8, 5120, 2, 1,  1, 16, 1, 16, 16, 16, 1, 16, 16, 8160, 0, 1024, 1, 0, 16, 0, 0, 0, 16, 16, 16, 0, 0, 0, 16, 16, 0, 0, 1, 16, 8, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 4094, 0]
case1_params = [40, 4096, 8, 40, 1,  40, 4096, 1, 4096, 4096, 4096, 1, 4096, 4096, 6080, 10, 1024, 40, 0, 4096, 1516, 1064, 4096, 1516, 1064, 0, 40, 320, 103, 103, 103, 79, 79, 79, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
case2_params = [48, 1, 1024, 48, 1, 1, 48, 1, 48, 48, 48, 1, 48, 48, 6080, 0, 1024, 48, 0, 1, 1, 1, 1, 1, 1, 0, 1, 48, 1, 1, 1, 1, 1, 1, 1024, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
case3_params = [40, 20, 44321, 20, 1,  1, 400, 1, 400, 400, 400, 1, 400, 400, 6080, 0, 1024, 40, 0, 10, 10, 10, 10, 10, 10, 0, 20, 400, 1, 1, 1, 1, 1, 1, 16376, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

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