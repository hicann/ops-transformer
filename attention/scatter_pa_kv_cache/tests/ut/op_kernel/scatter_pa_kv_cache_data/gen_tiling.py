#!/usr/bin/python
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


"""
struct ScatterPaKvCacheTilingData {
  int64_t usedCoreNum;
  int64_t blockFactor;
  int64_t tailBlockFactor;
  int64_t kHandleNumPerCore;
  int64_t vHandleNumPerCore;
  int64_t kTailHandleNum;
  int64_t vTailHandleNum;
  int64_t kLoopNum;
  int64_t vLoopNum;
  int64_t kHandleNumPerLoop;
  int64_t vHandleNumPerLoop;
  int64_t keyStride0;
  int64_t keyStride1;
  int64_t keyStride2;
  int64_t valueStride0;
  int64_t valueStride1;
  int64_t valueStride2;
  int64_t kHeadSize;
  int64_t vHeadSize;
  int64_t batch;
  int64_t numBlocks;
  int64_t blockSize;
  int64_t seqLen;
  int64_t numHead;
  int64_t numTokens;
  int64_t ubSize;
};
"""

params_info = {
    "test_scatter_pa_kv_cache_full_load1": [
        24,
        11,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        512,
        64,
        256,
        20,
        16,
        1,
        1,
        256,
        196608,
    ],
    "test_scatter_pa_kv_cache_full_load2": [
        24,
        11,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        512,
        64,
        256,
        20,
        16,
        1,
        1,
        256,
        196608,
    ],
    "test_case_nz_not_full_load_001": [
        24,
        11,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        512,
        64,
        256,
        20,
        16,
        1,
        1,
        256,
        196608,
    ],
    "test_case_nz_not_full_load_002": [
        24,
        11,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        512,
        64,
        256,
        20,
        16,
        1,
        1,
        256,
        196608,
    ],
}


def main():
    params_list = params_info[
        sys.argv[1]
    ]  # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == "__main__":
    main()
