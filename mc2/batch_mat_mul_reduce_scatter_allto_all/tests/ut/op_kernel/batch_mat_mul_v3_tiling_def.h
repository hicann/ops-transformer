/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_BATCH_MAT_MUL_V3_TILING_TILING_H_
#define _TEST_BATCH_MAT_MUL_V3_TILING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct Mc2L2cacheUseInfo{
    uint32_t l2CacheFlag;
};

struct Mc2L2cacheTilePara{
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};

struct Mc2MatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isHf32;
};

struct Mc2MultiBatchInfo {
    uint32_t batchUsedCoreNum;
    uint32_t aBatchDimAll;
    uint32_t bBatchDimAll;
    uint32_t cBatchDimAll;
    uint32_t aBatchDim0;
    uint32_t bBatchDim0;
    uint32_t cBatchDim0;
    uint32_t aBatchDim1;
    uint32_t bBatchDim1;
    uint32_t cBatchDim1;
    uint32_t aBatchDim2;
    uint32_t bBatchDim2;
    uint32_t cBatchDim2;
    uint32_t aBatchDim3;
    uint32_t bBatchDim3;
    uint32_t cBatchDim3;
    uint32_t iterBatch;
    uint32_t biasWithBatch;
    uint32_t mOri;
    uint32_t batchTileBlock;
    uint32_t aBatch;
    uint32_t bBatch;
};

struct Mc2MatmulV3TilingData {
    TCubeTiling matmulTiling;
    Mc2L2cacheTilePara tileL2cacheTiling;
    Mc2MatMulRunInfo matmulRunInfo;
    Mc2L2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};

struct Mc2BatchMatmulTilingData {
    Mc2MatmulV3TilingData matmulTiling;
    Mc2MultiBatchInfo Mc2multiBatchInfo;
};

inline void InitMatmulTilingData(uint8_t* tiling, Mc2BatchMatmulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(Mc2BatchMatmulTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                       \
    Mc2BatchMatmulTilingData tiling_data;                                              \
    InitMatmulTilingData(tiling_arg, &tiling_data)
#endif  // _TEST_BATCH_MAT_MUL_V3_TILING_TILING_H_
