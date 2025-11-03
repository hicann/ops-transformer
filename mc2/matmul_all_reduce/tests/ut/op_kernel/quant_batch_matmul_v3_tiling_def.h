/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_tiling_def.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V3_TILING_DEF_H
#define QUANT_BATCH_MATMUL_V3_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

struct Mc2L2cacheTileParam
{
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
    uint32_t isBasicTiling = 0;
};

struct QuantBatchMatmulV3Params
{
    uint32_t batchA = 0;
    uint32_t batchB = 0;
    uint32_t batchC = 0;
    uint32_t batchA1 = 0;
    uint32_t batchA2 = 0;
    uint32_t batchA3 = 0;
    uint32_t batchA4 = 0;
    uint32_t batchB1 = 0;
    uint32_t batchB2 = 0;
    uint32_t batchB3 = 0;
    uint32_t batchB4 = 0;
    uint32_t batchC1 = 0;
    uint32_t batchC2 = 0;
    uint32_t batchC3 = 0;
    uint32_t batchC4 = 0;
    uint32_t singleCoreBatch = 0;
    uint32_t isPerTensor = 0;
    uint32_t isPertoken = 0;
    uint32_t isDoubleScale = 0;
    uint32_t biasThreeDim = 0;  // 整块的个数
    uint32_t ubCalcM = 0;
    uint32_t ubCalcN = 0;
    uint32_t needUbBuffer = 0;
    uint32_t realSingleCoreM = 0;
    uint32_t realSingleCoreN = 0;
    uint32_t biasDtype = 0;
    uint32_t ubSize = 0;
    uint32_t isMClash = 0;
    uint32_t isNClash = 0;
    uint32_t groupSizeM = 0;
    uint32_t groupSizeN = 0;
    uint32_t groupSizeK = 0;
};

struct Mc2SlidingWindowParam {
    uint32_t mTailTile = 0;
    uint32_t nTailTile = 0;
};

struct QuantBatchMatmulV3TilingData
{
    QuantBatchMatmulV3Params params;
    TCubeTiling matmulTiling;
    Mc2L2cacheTileParam tileL2cacheTiling;
    Mc2SlidingWindowParam adaptiveSlidingWin;
};

inline void InitQuantBatchMatmulV3TilingData(uint8_t* tiling, QuantBatchMatmulV3TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(QuantBatchMatmulV3TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    QuantBatchMatmulV3TilingData tiling_data;                                                 \
    InitQuantBatchMatmulV3TilingData(tiling_arg, &tiling_data)
#endif  // QUANT_BATCH_MATMUL_V3_TILING_DEF_H
