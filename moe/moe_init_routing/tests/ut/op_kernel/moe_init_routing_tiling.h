/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MOE_INIT_ROUTING_TILING_H__
#define __MOE_INIT_ROUTING_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct VBSComputeTilingData {
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
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, VBSComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(VBSComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, VBSComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(VBSComputeTilingData));
}
#endif

#pragma pack(1)
struct VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, VMSMiddleComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(VMSMiddleComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, VMSMiddleComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(VMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, SortOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SortOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, SortOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(SortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t activateRows = 0;
    int64_t perCoreRows = 0;
    int64_t perCoreK = 0;
    int64_t perCorePerLoopK = 0;
    int64_t perCoreLastLoopK = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCoreK = 0;
    int64_t lastCorePerLoopK = 0;
    int64_t lastCoreLastLoopK = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t maxColsOneLoop = 0;
    int64_t splitFlag = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, GatherOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(GatherOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, GatherOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(GatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingTilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t cols = 0;
    int64_t k = 0;
    VBSComputeTilingData vbsComputeParamsOp;
    VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    SortOutComputeTilingData sortOutComputeParamsOp;
    GatherOutComputeTilingData srcToDstComputeParamsOp;
    GatherOutComputeTilingData gatherOutComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeInitRoutingTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeInitRoutingTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeInitRoutingTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeInitRoutingTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeInitRoutingTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif
