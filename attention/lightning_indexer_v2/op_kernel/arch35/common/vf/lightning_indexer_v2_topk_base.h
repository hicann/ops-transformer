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
 * \file lightning_indexer_v2_topk_base.h
 * \brief Common TopK helpers shared by LightningIndexer variants.
 */

#ifndef LIGHTNING_INDEXER_V2_TOPK_BASE_H
#define LIGHTNING_INDEXER_V2_TOPK_BASE_H

#include "../lightning_indexer_v2_base_arch35.h"

namespace liV2TopkCommon {

constexpr uint32_t RADIX_BUFFER_SIZE = 256;
constexpr uint32_t B32_RADIX_BUFFER_NUM = 5;
constexpr uint32_t B16_RADIX_BUFFER_NUM = 3;

// Index storage stays uint32_t; only temporary gather indices use IndexT.
template <typename IndexT, uint32_t RadixBuffers>
__aicore__ inline uint32_t GetGatherTmpBufferSize(uint32_t topK, uint32_t trunkLen)
{
    uint64_t bufferSize1 =
        (2 * LIV2Common::Align(topK, RADIX_BUFFER_SIZE) + RadixBuffers * RADIX_BUFFER_SIZE + 64) * sizeof(uint32_t);
    uint64_t bufferSize2 = (LIV2Common::Align(topK, RADIX_BUFFER_SIZE) + trunkLen) * sizeof(IndexT);
    uint64_t reuseBufferSize = LIV2Common::Align(topK, RADIX_BUFFER_SIZE) * sizeof(uint32_t);
    return bufferSize1 + bufferSize2 - reuseBufferSize;
}

__aicore__ inline uint32_t GetGatherLoopOffset(uint32_t topK, uint32_t trunkLen, uint32_t loopIdx)
{
    return topK < trunkLen ? loopIdx * trunkLen - LIV2Common::Align(topK, RADIX_BUFFER_SIZE) : (loopIdx - 1) * trunkLen;
}

} // namespace liV2TopkCommon

#endif // LIGHTNING_INDEXER_V2_TOPK_BASE_H
