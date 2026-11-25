/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_UB_PLAN_ARCH92_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_UB_PLAN_ARCH92_H

#include <cstdint>

namespace SligKlLoss {

__aicore__ inline constexpr uint32_t GetSligReduceSumBufferBytes(uint32_t topK, uint64_t keyTokenCount)
{
    // Clamp the physical TND/BSND extent before narrowing it; TND T2 is a 64-bit field.
    const uint32_t maxReduceElements = keyTokenCount < topK ? static_cast<uint32_t>(keyTokenCount) : topK;
    // V1 writes four complete FP32 registers even for a one-element tail.
    constexpr uint32_t reduceWriteElements = 256;
    const uint32_t alignedElements =
        (maxReduceElements + reduceWriteElements - 1) / reduceWriteElements * reduceWriteElements;
    return (alignedElements == 0 ? reduceWriteElements : alignedElements) * sizeof(float);
}

// The 16-row layout is enabled only after reclaiming at least 30 KiB from reduceSum.
__aicore__ inline constexpr uint32_t GetSligGatherRows(uint32_t reduceSumBytes)
{
    return reduceSumBytes <= 2048 ? 16 : 8;
}

__aicore__ inline constexpr uint32_t GetSligGatherInputBufferBytes(uint32_t gatherRows, bool hasRope)
{
    // Rope LoadAlign still reads 128 halfwords from a logical 64-column row.
    return 2 * gatherRows * (512 + (hasRope ? 64 : 0)) * sizeof(uint16_t) + (hasRope ? 128 : 0);
}

__aicore__ inline constexpr uint32_t GetSligGatherGradBufferBytes(uint32_t gatherRows, uint32_t gradBytes)
{
    const uint32_t gatherBytes = 2 * gatherRows * 512 * sizeof(uint16_t);
    const uint32_t sharedBytes = gatherBytes > 18 * 1024 ? gatherBytes : 18 * 1024;
    return sharedBytes > gradBytes ? sharedBytes : gradBytes;
}

__aicore__ inline constexpr bool CanUseSligSingleQueryDKeyOutput(uint32_t batchSize, uint32_t maxQueryLength,
                                                                 uint32_t maxKeyLength, int64_t totalQuerySize,
                                                                 uint32_t topK, int32_t pKBaseSize)
{
    // max(Q)=1 and sum(Q)=B exclude empty and multi-query batches without using physical T1.
    return batchSize > 0 && maxQueryLength == 1 && totalQuerySize == static_cast<int64_t>(batchSize) &&
           maxKeyLength > 0 && maxKeyLength <= 128 && maxKeyLength <= topK && pKBaseSize > 0 &&
           maxKeyLength <= static_cast<uint32_t>(pKBaseSize);
}

} // namespace SligKlLoss
#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_UB_PLAN_ARCH92_H
