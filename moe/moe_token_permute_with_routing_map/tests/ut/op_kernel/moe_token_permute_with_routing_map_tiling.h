/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_TILING_H_
#define _MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define DTYPE_PERMUTE_PROBS bfloat16_t

#define __aicore__

struct PermuteVBSComputeRMTilingData {
    int64_t needCoreNum = 1;
    int64_t perCoreElements = 8;
    int64_t perCoreLoops = 1;
    int64_t perCorePerLoopElements = 8;
    int64_t perCoreLastLoopElements = 8;
    int64_t lastCoreElements = 8;
    int64_t lastCoreLoops = 1;
    int64_t lastCorePerLoopElements = 8;
    int64_t lastCoreLastLoopElements = 8;
    int64_t oneLoopMaxElements = 6112;
    int64_t lastCoreWSindex = 0;
    int64_t frontcoreTask = 1;
    int64_t tailcoreTask = 1;
    int64_t frontCoreNum = 1;
    int64_t tailCoreNum = 0;
};

struct PermuteVMSMiddleComputeRMTilingData {
    int64_t needCoreNum = 0;
};

struct PermuteSortOutComputeRMTilingData {
    int64_t oneLoopMaxElements = 1024;
};

struct IndexCopyComputeRMTilingData {
    int64_t needCoreNum = 4;
    int64_t frontCoreNum = 4;
    int64_t tailCoreNum = 0;
    int64_t coreCalcNum = 1;
    int64_t coreCalcTail = 0;
    int64_t oneTokenBtypeSize = 6;
    int64_t onceIndicesTokenMoveTimes = 2;
    int64_t onceUbTokenNums = 2454;
    int64_t onceIndicesTokenNums = 4908;
    int64_t onceIndices = 9816;
    int64_t oneTokenlastMove = 1;
    int64_t oneTokenOnceMove = 1;
    int64_t oneTokenMoveTimes = 1;
    int64_t frontCoreLoop = 1;
    int64_t frontCoreLastTokenNums = 1;
    int64_t tailCoreLoop = 0;
    int64_t tailCoreLastTokenNums = 4908;
    int64_t tailLastonceIndicesTokenMoveTimes = 2;
    int64_t tailLastIndicesLastTokenNums = 2454;
    int64_t frontLastonceIndicesTokenMoveTimes = 1;
    int64_t frontLastIndicesLastTokenNums = 1;
    int64_t numOutTokens = 8;
    int64_t tokenUB = 78528;
    int64_t indicesUB = 9824;
};

struct MaskedSelectRMTilingData {
    uint64_t formerNum = 4;
    uint64_t formerLength = 1;
    uint64_t formertileNum = 1;
    uint64_t formertileLength = 1;
    uint64_t formerlasttileLength = 1;
    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailtileNum = 0;
    uint64_t tailtileLength = 0;
    uint64_t taillasttileLength = 1;
    uint64_t tokenNum = 8;
    int64_t needCoreNum = 1;
};

struct MoeTokenPermuteWithRoutingMapTilingData {
    int64_t coreNum = 1;
    int64_t n = 4;
    int64_t cols = 3;
    int64_t colsAlign = 16;
    int64_t topK = 2;
    int64_t expertNum = 2;
    int64_t capacity = 2;
    int64_t hasProb = 0;
    PermuteVBSComputeRMTilingData vbsComputeParamsOp;
    PermuteVMSMiddleComputeRMTilingData vmsMiddleComputeParamsOp;
    PermuteSortOutComputeRMTilingData sortOutComputeParamsOp;
    IndexCopyComputeRMTilingData indexCopyComputeParamsOp;
    MaskedSelectRMTilingData maskedSelectParamsOp;
};

#define GET_TILING_DATA(tilingData, tilingPointer)      \
    MoeTokenPermuteWithRoutingMapTilingData tilingData;
#endif