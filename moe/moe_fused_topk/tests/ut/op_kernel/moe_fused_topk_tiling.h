/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_MOE_FUSED_TOPK_TILING_H__
#define __TEST_MOE_FUSED_TOPK_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeFusedTopkTilingData {
    uint32_t firstDimSize = 0;
    uint32_t secondDimSize = 0;
    uint32_t addNumDimSize = 0;
    uint32_t groupNum = 0;
    uint32_t groupTopk = 0;
    uint32_t topN = 0;
    uint32_t topK = 0;

    uint32_t activateType = 0;
    uint32_t isNorm = 0;
    float scale = 0;
    uint32_t groupEles = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t batchPerCore = 0;
    uint32_t tailBatch = 0;

    uint32_t expertNum = 0;
    uint32_t tableDim = 0;
    uint32_t topkMaxValue = 0;
    uint32_t topkMinValue = 0;
    uint32_t reserved = 0;
    uint64_t workspacePerCore = 0;
    TopkTiling topkTilingData;
};
#pragma pack()

inline void InitMoeFusedTopkTilingData(uint8_t *tiling, MoeFusedTopkTilingData *const_data)
{
    uint32_t *src = (uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeFusedTopkTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeFusedTopkTilingData tiling_data; \
    InitMoeFusedTopkTilingData(tiling_arg, &tiling_data)

#endif // __TEST_MOE_FUSED_ADD_TOPK_TILING_H__
