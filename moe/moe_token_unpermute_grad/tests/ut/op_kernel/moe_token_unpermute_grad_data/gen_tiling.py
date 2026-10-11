#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import sys

# [10, 3, 64] bfloat16_t
case0_params = [10, 3, 64, 30, 10, 38, 1, 0, 3, 0, 256, 1, 64, 3, 3, 16, 196352]
case1_params = [10, 1, 64, 10, 10, 38, 1, 0, 1, 0, 64, 1, 64, 1, 1, 16, 196352]
case2_params = [10, 3, 8192, 30, 10, 38, 1, 0, 3, 0, 7936, 2, 256, 1, 3, 16, 196352]

params_info = {"case0": case0_params, "case1": case1_params, "case2": case2_params}


def main():
    params_list = params_info[
        sys.argv[1]
    ]  # python gen_tiling.py case0  sys.argv[1]="case0"
    base_params = np.array(params_list, dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == "__main__":
    main()
