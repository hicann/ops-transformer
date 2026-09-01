/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_

#ifndef __CCE_AICORE__
#include <cstddef>
#include <cstdint>
#endif

#include "kernel_tiling/kernel_tiling.h"

namespace QkvRmsNormRopeCacheWithKScaleKernelTiling {

#pragma pack(push, 8)
struct QkvRmsNormRopeCacheWithKScaleTilingData {
    uint64_t totalTokens = 0;
    uint64_t batch = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t headDim = 0;
    uint64_t blockSize = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t kvCacheStrideBlock = 0;
    uint64_t kvCacheStrideHead = 0;
    uint64_t kvCacheStrideToken = 0;
    uint64_t kScaleCacheStrideBlock = 0;
    uint64_t kScaleCacheStrideHead = 0;
    uint64_t kScaleCacheStrideToken = 0;
    uint64_t tokenTile = 0;
    float epsilon = 0.0F;
    // M-RoPE uses only the H/W lane capacities at device side.  The T
    // capacity is validated by host tiling but intentionally is not consumed
    // by the kernel.
    uint64_t mropeSectionH = 0;
    uint64_t mropeSectionW = 0;
};
#pragma pack(pop)

static_assert(sizeof(QkvRmsNormRopeCacheWithKScaleTilingData) == 144U,
              "QkvRmsNormRopeCacheWithKScaleTilingData ABI size changed");
static_assert(__builtin_offsetof(QkvRmsNormRopeCacheWithKScaleTilingData, tokenTile) == 112U,
              "tokenTile ABI offset changed");
static_assert(__builtin_offsetof(QkvRmsNormRopeCacheWithKScaleTilingData, epsilon) == 120U,
              "epsilon ABI offset changed");
static_assert(__builtin_offsetof(QkvRmsNormRopeCacheWithKScaleTilingData, mropeSectionH) == 128U,
              "mropeSectionH ABI offset changed");

} // namespace QkvRmsNormRopeCacheWithKScaleKernelTiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_
