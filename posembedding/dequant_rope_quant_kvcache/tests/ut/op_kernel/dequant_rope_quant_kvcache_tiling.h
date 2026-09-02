/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_DEQUANT_ROPE_QUANT_KVCACHE_H
#define TEST_DEQUANT_ROPE_QUANT_KVCACHE_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define DTYPE_COS half
#define DTYPE_X half

struct DequantRopeQuantKvcacheTilingData {
    int64_t qHeadNum = 8;
    int64_t kvHeadNum = 1;
    int64_t hiddenSize = 128;
    int64_t hiddenSizeFp32Align = 128 * 4;
    int64_t hiddenSizeFp16Align = 128 * 2;
    int64_t hiddenSizeInt8Align = 128;
    int64_t OnceUBMaxS = 15;
    int64_t cacheSeqlen = 1024;
    int64_t seqlen = 1;
    int64_t qHiddenSize = 128 * 8;
    int64_t kHiddenSize = 128;
    int64_t vHiddenSize = 128;
    int64_t realCoreNum = 40;
    int64_t frontCoreNum = 30;
    int64_t blockFactor = 3;
    int64_t tailCoreBlockFactor = 10;
    int64_t hasQuantOffset = 1;
    int64_t ifKVout = 1;
    int64_t isPA = 1;
    int64_t hasBias = 1;
    int64_t hasAS = 1;
};

inline void IDequantRopeQuantKvcacheTilingData(uint8_t *tiling, DequantRopeQuantKvcacheTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(DequantRopeQuantKvcacheTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    DequantRopeQuantKvcacheTilingData tilingData; \
    IDequantRopeQuantKvcacheTilingData(tilingPointer, &tilingData)
#endif
