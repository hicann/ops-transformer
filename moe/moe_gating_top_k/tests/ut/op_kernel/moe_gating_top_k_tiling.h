/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_tiling.h
 * \brief
 */

#ifndef __MOE_INIT_ROUTING_TILING_H__
#define __MOE_INIT_ROUTING_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeGatingTopKTilingData {
    int64_t needCoreNum = 0;
    int64_t rowCount = 0;
    int64_t perCoreRowCount = 0;
    int64_t lastCoreRowCount = 0;
    int64_t expertCount = 0;
    int64_t addBias;
    int64_t k = 0;
    int64_t kGroup = 0;
    int64_t groupCount = 0;
    int64_t perGroupExpertCount = 0;
    int64_t perGroupExpertCountAlign = 0;
    int64_t groupSelectMode = 0;
    int64_t renorm = 0;
    int64_t normType = 0;
    int64_t outFlag = 0;
    int64_t vmsCount = 0;
    float routedScalingFactor = 0;
    float eps = 0;
    int64_t calTmpBufUbSize = 0;
};
#pragma pack()

#pragma pack(1)
struct MoeGatingTopKRegbaseTilingData {
    int64_t needCoreNum = 0;
    int64_t rowCount = 0;
    int64_t perCoreRowCount = 0;
    int64_t lastCoreRowCount = 0;
    int64_t expertCount = 0;
    int64_t addBias;
    int64_t k = 0;
    int64_t kGroup = 0;
    int64_t groupCount = 0;
    int64_t perGroupExpertCount = 0;
    int64_t perGroupExpertCountAlign = 0;
    int64_t groupSelectMode = 0;
    int64_t renorm = 0;
    int64_t normType = 0;
    int64_t y2Flag = 0;
    int64_t vmsCount = 0;
    float routedScalingFactor = 0;
    float eps = 0;
    int64_t calTmpBufUbSize = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKTilingData));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKRegbaseTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKRegbaseTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKRegbaseTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKRegbaseTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeGatingTopKTilingData tiling_data;         \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif