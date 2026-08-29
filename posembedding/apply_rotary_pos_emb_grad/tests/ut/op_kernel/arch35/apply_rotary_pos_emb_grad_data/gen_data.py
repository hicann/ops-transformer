#!/usr/bin/python3
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


def gen_data_and_golden(a, b, c, d, dtype_q):
    # 输入梯度与前向输入: [B, S, N, D]; cos/sin: [B, S, 1, D] (AB/SBND 模板, cos 的 B 轴与 grad 一致,
    # kernel 按 (b,s) 行访问 cos, 见 op_host ValidateBroadcastByLayout: cosB==gqB → AB 模板)
    data_x = np.random.uniform(-2, 2, [int(a), int(b), int(c), int(d)]).astype(dtype_q)
    data_y = np.random.uniform(-2, 2, [int(a), int(b), int(c), int(d)]).astype(dtype_q)
    data_cos = np.random.uniform(-3, 4, [int(a), int(b), 1, int(d)]).astype(dtype_q)
    data_sin = np.random.uniform(-4, 5, [int(a), int(b), 1, int(d)]).astype(dtype_q)
    data_dx = np.random.uniform(-2, 2, [int(a), int(b), int(c), int(d)]).astype(dtype_q)
    data_dy = np.random.uniform(-2, 2, [int(a), int(b), int(c), int(d)]).astype(dtype_q)
    data_dx.tofile("./dx.bin")
    data_dy.tofile("./dy.bin")
    data_cos.tofile("./cos.bin")
    data_sin.tofile("./sin.bin")
    data_x.tofile("./x.bin")
    data_y.tofile("./y.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    exit(0)
