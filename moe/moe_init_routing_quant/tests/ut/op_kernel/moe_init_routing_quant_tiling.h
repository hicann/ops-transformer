/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_quant_tiling.h
 * \brief
 */
#ifndef __MOE_INIT_ROUTING_QUANT_TILING_H__
#define __MOE_INIT_ROUTING_QUANT_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct QuantVBSComputeTilingData {
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
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QuantVBSComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(QuantVBSComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, QuantVBSComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QuantVBSComputeTilingData));
}
#endif

#pragma pack(1)
struct QuantVMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QuantVMSMiddleComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(QuantVMSMiddleComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, QuantVMSMiddleComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QuantVMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct QuantSortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QuantSortOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(QuantSortOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, QuantSortOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QuantSortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct QuantGatherOutComputeTilingData {
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
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QuantGatherOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(QuantGatherOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, QuantGatherOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QuantGatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingQuantTilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t cols = 0;
    int64_t k = 0;
    float scale = 0;
    float offset = 0;
    QuantVBSComputeTilingData vbsComputeParamsOp;
    QuantVMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    QuantSortOutComputeTilingData sortOutComputeParamsOp;
    QuantGatherOutComputeTilingData srcToDstComputeParamsOp;
    QuantGatherOutComputeTilingData gatherOutComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeInitRoutingQuantTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeInitRoutingQuantTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeInitRoutingQuantTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeInitRoutingQuantTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeInitRoutingQuantTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif
