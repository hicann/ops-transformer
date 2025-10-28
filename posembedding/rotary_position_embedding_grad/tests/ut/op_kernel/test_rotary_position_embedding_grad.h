/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct RopeHalfGradUTParams {
    int64_t layout;
    int64_t xShapeSize;
    int64_t cosShapeSize;
    int64_t dimB;
    int64_t dimS;
    int64_t dimN;
    int64_t dimD;
    int64_t cosDimB;
    int64_t cosDimN;
    int64_t halfDimDAlignNum;

    int64_t coreData;
    int64_t coreLast;
    int64_t copyLoop;
    int64_t copyTail;
    int64_t lastCopyLoop;
    int64_t lastCopyTail;
    int64_t alignUbSize;
    int64_t calcUbSize;
    int64_t coreUsed;
    int64_t coreNum;

    int64_t firstReduce;
    int64_t secondReduce;
    int64_t ubLoopGap;
    uint64_t blockLenInner;
    uint64_t strideInner;
    uint64_t blockLenPadInner;
    uint64_t stridePadInner;
};

struct RopeInterleavedGradUTParams {
    uint64_t batchSize = 1;
    uint64_t seqLen = 64;
    uint64_t numHeads = 2;
    uint64_t headDim = 62;
    uint64_t alignHeadDim = 64;
    uint64_t padHeadDim = 2;
    uint64_t frontCoreNum = 16;
    uint64_t tailCoreNum = 32;
    uint64_t seqFrontLen = 2;
    uint64_t seqTailLen = 1;
    uint64_t seqFrontCalcNum = 2;
    uint64_t seqFrontCalcLoop = 1;
    uint64_t seqFrontCalcTail = 0;
    uint64_t seqTailCalcNum = 1;
    uint64_t seqTailCalcLoop = 1;
    uint64_t seqTailCalcTail = 0;
    uint64_t numHeadsLength = 0;
    uint64_t numHeadsLoop = 0;
    uint64_t numHeadsTail = 0;
    uint64_t batchNumHeadsLength = 0;
    uint64_t batchNumHeadsLoop = 0;
    uint64_t batchNumHeadsTail = 0;
    uint64_t layout = 0;
};

struct RotaryPositionEmbeddingGradUTTilingData {
    RopeHalfGradUTParams ropeHalfGradParams;
    RopeInterleavedGradUTParams ropeInterleavedGradParams;
};

inline void InitRotaryPositionEmbeddingGradTilingData(
    uint8_t* tiling, RotaryPositionEmbeddingGradUTTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(RotaryPositionEmbeddingGradUTTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)    \
    RotaryPositionEmbeddingGradUTTilingData tilingData; \
    InitRotaryPositionEmbeddingGradTilingData(tilingPointer, &tilingData)
#endif