# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

#!/usr/bin/python3
import sys
import os
import numpy as np
import re

def gen_data(m, n, k,d_type="float16"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32
    }
    np_type = d_type_dict[d_type]
    x1 = np.random.uniform(-1 , 1, [int(m), int(k)]).astype(np_type)
    x2 = np.random.uniform(-1 , 1, [int(k), int(n)]).astype(np_type)

    x1.tofile(f"{d_type}_input_matmul_all_reduce_0.bin")
    x2.tofile(f"{d_type}_input_matmul_all_reduce_1.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data(sys.argv[1], sys.argv[2], sys.argv[3])
