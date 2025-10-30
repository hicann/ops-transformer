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

#           N   K  H     active_rows capacity  expert_num  drop_pad_mode expert_tokens_count_flag expert_tokens_capacity_flag
#  case0:   8   6  30    -1          6         8           0             1                        false
#  case1:   320 56 3000  -1          200       32          1             0                        true
#  case2:   8   6  30    32          -1        8           0             1                        false

# coreNum, n, cols, k

case0_params = [64, 1, 30, 1, 6, 8, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8160, 0, 2040, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 30, 30, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 30, 30, 1]
case1_params = [40, 320, 30000, 56 ,200, 32, 1, 0, 1, 2, 4, 4480, 1, 4480, 4480, 4480, 1, 4480, 4480, 6144, 0, 1024, 40 ,0, 448, 448, 448, 448, 448, 448, 0, 0, 0, 0, 0, 40, 0, 448, 448, 448, 448, 448, 448, 1, 1, 30000, 30000, 1, 40, 17920, 448, 448, 448, 448, 448, 448, 1, 1, 9017, 2949, 4]
case2_params = [40, 8, 30, 6, 0, 8, 0, 1, 0, 0, 1, 48, 1, 48, 48, 48, 1, 48, 48, 6144, 0, 1024, 24, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 24, 0, 2, 2, 2, 2, 2, 2, 1, 1, 30, 30, 1, 24, 32, 2, 2, 2, 2, 2, 2, 1, 1, 30, 30, 1]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()