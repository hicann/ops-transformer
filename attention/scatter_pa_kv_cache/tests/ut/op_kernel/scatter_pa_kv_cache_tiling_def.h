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
 * \file test_scatter_pa_kv_cache.h
 * \brief
 */
#ifndef _SCATTER_PA_KV_CACHE_H_
#define _SCATTER_PA_KV_CACHE_H_

#include <cstdint>
#include <cstring>

#define DTYPE_KEY half
#define DTYPE_VALUE half

struct ScatterPaKvCacheTilingData {
    int64_t usedCoreNum;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t kHandleNumPerCore;
    int64_t vHandleNumPerCore;
    int64_t kTailHandleNum;
    int64_t vTailHandleNum;
    int64_t kLoopNum;
    int64_t vLoopNum;
    int64_t kHandleNumPerLoop;
    int64_t vHandleNumPerLoop;
    int64_t keyStride0;
    int64_t keyStride1;
    int64_t keyStride2;
    int64_t valueStride0;
    int64_t valueStride1;
    int64_t valueStride2;
    int64_t kHeadSize;
    int64_t vHeadSize;
    int64_t batch;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t seqLen;
    int64_t numHead;
    int64_t numTokens;
    int64_t ubSize;
    uint64_t kStride;
    uint64_t vStride;
    uint64_t kOffset;
    uint64_t vOffset;
};

inline void InitScatterPaKvCacheTilingData(uint8_t *tiling, ScatterPaKvCacheTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(ScatterPaKvCacheTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    ScatterPaKvCacheTilingData tilingData; \
    InitScatterPaKvCacheTilingData(tilingPointer, &tilingData)
#endif
