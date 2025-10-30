/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MOE_INIT_ROUTING_V2_GRAD_TILING_H__
#define __MOE_INIT_ROUTING_V2_GRAD_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeV2GradComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t elementCopyLoops = 0;
    int64_t elementPerCopyCols = 0;
    int64_t elementLastCopyCols = 0;
    int64_t binaryAddBufferNum = 0;
    int64_t tmpBufferNum = 0;
    int64_t exponentOfBinary = 0;
    int64_t copyBufferSize = 0;
    int64_t tokensFormer = 0;
    int64_t perCoreTokensLoop = 0;
    int64_t perCoreTailTokensFormer = 0;
    int64_t lastCoreTokensLoop = 0;
    int64_t lastCoreTailTokensFormer = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeV2GradComputeTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV2GradComputeTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeV2GradComputeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV2GradComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingV2GradTilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t e = 0;
    int64_t c = 0;
    int64_t cols = 0;
    int64_t k = 0;
    int64_t activeNum = 0;
    MoeV2GradComputeTilingData MoeV2GradComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeInitRoutingV2GradTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeInitRoutingV2GradTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeInitRoutingV2GradTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeInitRoutingV2GradTilingData));
}
#endif


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
tiling_struct tiling_data; \
InitTilingData(tiling_arg, &tiling_data)


#define GET_TILING_DATA(tiling_data, tiling_arg) \
MoeInitRoutingV2GradTilingData tiling_data; \
InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_GRAD_EXPANDED_X float

#endif