/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MOE_INIT_ROUTING_QUANT_V2_TILING_H__
#define __MOE_INIT_ROUTING_QUANT_V2_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeV2VBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV2VBSComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV2VBSComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV2VBSComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV2VBSComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV2VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV2VMSMiddleComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV2VMSMiddleComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV2VMSMiddleComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV2VMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV2SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV2SortOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV2SortOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV2SortOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV2SortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV2GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t activateRows = 0;
    int64_t perCoreRows = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t perCoreLoops = 0;
    int64_t lastCoreLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
    int64_t colLoops = 0;
    int64_t scatterMode = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV2GatherOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV2GatherOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV2GatherOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV2GatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingV2TilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t cols = 0;
    int64_t k = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = 0;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t expertTokensBeforeCapacityFlag = 0;
    MoeV2VBSComputeTilingData vbsComputeParamsOp;
    MoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV2SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    MoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeInitRoutingV2TilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeInitRoutingV2TilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeInitRoutingV2TilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeInitRoutingV2TilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeInitRoutingV2TilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif
