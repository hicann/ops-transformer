/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __QUANT_MATMUL_DEQUANT_TILING_H__
#define __QUANT_MATMUL_DEQUANT_TILING_H__

#include "kernel_tiling/kernel_tiling.h"

#define ORIG_DTYPE_WEIGHT_SCALE DT_FLOAT
#define ORIG_DTYPE_X_SCALE DT_FLOAT

#if defined(__CCE_KT_TEST__)
#include <cstdint>
#include <cstring>
#endif

#pragma pack(1)
struct QuantMatmulDequantTilingData {
    uint32_t CoreNum = 0;
    uint32_t perToken = 0;
    uint32_t dynamicQuant = 0;
    uint32_t smoothScale = 0;

    uint32_t originE = 0;
    uint32_t originM = 0;
    uint32_t originN = 0;
    uint32_t originK = 0;
    uint32_t originKAligned32 = 0;
    uint32_t originKAligned512 = 0;
    uint32_t fracN = 0;
    uint32_t fracK = 0;

    uint32_t dynamicBaseK = 0;
    uint32_t dynamicIterK = 0;
    uint32_t dynamicBaseKTail = 0;

    uint32_t singleCoreFracN = 0;
    uint32_t singleCoreFracNTail = 0;
    uint32_t baseFracN = 0;
    uint32_t baseFracK = 0;
    uint32_t baseFracNL0C = 0;
    uint32_t ubBaseK = 0;
    uint32_t ubIterK = 0;
    uint32_t ubBaseKTail = 0;

    uint32_t fracM = 0;
    uint32_t tailM = 0;
    uint32_t processXKBaseNMax = 0;
    uint32_t processXKBaseN = 0;
    uint32_t processXKloop = 0;
    uint32_t processXKloopPerfracM = 0;
    uint32_t processXKTailN = 0;
    uint32_t MMmod = 0;
    uint32_t MCoreNum = 0;
    uint32_t NCoreNum = 0;
    uint32_t singleCoreM = 0;
    uint32_t singleCoreN = 0;
    uint32_t singleCoreMTail = 0;
    uint32_t singleCoreNTail = 0;
    uint32_t baseMNum = 0;
    uint32_t baseNNum = 0;
    uint32_t baseKNum = 0;

    uint32_t swiftGEMVThreshold = 0;
    uint32_t baseNNum_2 = 0;
    uint32_t baseKNum_2 = 0;
    uint32_t baseK = 0;
    uint32_t baseKTail = 0;
    uint32_t baseK_2 = 0;
    uint32_t baseKTail_2 = 0;

    uint64_t ubKMask = 0;
    uint32_t isXScaleHalf = 0;
};
#pragma pack()

#if defined(__CCE_KT_TEST__)
inline void InitTilingData(uint8_t *tiling, QuantMatmulDequantTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QuantMatmulDequantTilingData));
}
#else
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QuantMatmulDequantTilingData *const_data)
{
    for (auto i = 0; i < sizeof(QuantMatmulDequantTilingData) / 4; i++) {
        *(int32_t *)((int32_t *)const_data + i) = *((__gm__ int32_t *)tiling + i);
    }
}
#endif

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    QuantMatmulDequantTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)
#endif
