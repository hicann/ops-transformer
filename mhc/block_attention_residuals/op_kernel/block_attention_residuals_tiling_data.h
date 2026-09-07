/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_attention_residuals_tiling_data.h
 * \brief BlockAttentionResiduals tiling data structure
 */
#ifndef ATTN_RES_FWD_TILING_DATA_H
#define ATTN_RES_FWD_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace BlockAttentionResiduals {
#define BLOCK_ATTENTION_RESIDUALS_TILING_ALIGN 8
#pragma pack(push, BLOCK_ATTENTION_RESIDUALS_TILING_ALIGN)
struct alignas(BLOCK_ATTENTION_RESIDUALS_TILING_ALIGN) BlockAttentionResidualsTilingData {
    int64_t numTokens;
    int64_t numBlocks;
    int64_t validBlockNum;
    int64_t hiddenSize;
    int64_t tokensPerCore;
    uint32_t usedCoreNum;
    float normEps;
    float invHiddenSize;
    uint64_t wsSizePerToken;
    int64_t blockCount;
    uint32_t needBackward;   // 0/1
    uint32_t stagingBytes;   // 512 倍数；false 时为 0
    uint32_t tokensPerFlush; // staging 可攒 token 数；false 时为 0
    uint32_t elemsPerToken;  // 2*blockCount；false 时可写 0
    int64_t hiddenSizeChunk; // HSLICE 分支的 h 切块大小；非 HSLICE 时可为 0
};
#pragma pack(pop)
#undef BLOCK_ATTENTION_RESIDUALS_TILING_ALIGN
} // namespace BlockAttentionResiduals

#endif // ATTN_RES_FWD_TILING_DATA_H
