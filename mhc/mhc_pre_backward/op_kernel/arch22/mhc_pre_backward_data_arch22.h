/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_backward_data_arch22.h
 * \brief tiling data struct for arch22 (A2)
 */

#ifndef MHC_PRE_BACKWARD_DATA_ARCH22_H
#define MHC_PRE_BACKWARD_DATA_ARCH22_H

#include <cstdint>

struct MhcPreBackwardArch22TilingData {
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t fusionSize = 0; // phi 第0维: factN + 2*N (A2: N!+2N, A3: N²+2N)
    int64_t isBSNN = 0;     // gradHRes 布局: 1=BSNN(B,S,N,N), 0=BSN!(B,S,N!)
    int64_t c0 = 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    int64_t ubSize = 0;
    int64_t withGamma = 0;
    int64_t withGradXPost = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

#endif // MHC_PRE_BACKWARD_DATA_ARCH22_H
