/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_BATCH_LOSS_ARCH92_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_BATCH_LOSS_ARCH92_H

#include <cstdint>

namespace SligBatchLoss {
constexpr uint32_t SLOT_BYTES = 32;
constexpr uint32_t SLOT_FLOATS = SLOT_BYTES / sizeof(float);
// V8 reuses its 32 KiB input queue to merge the batch partials.
constexpr uint32_t MAX_SLOTS = 32 * 1024 / SLOT_BYTES;

#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr int64_t
    SegmentSlots(int64_t batches, int64_t owners)
{
    return batches + owners - 1;
}

#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr uint32_t
    SlotCount(int64_t queries, int64_t maxQueryLen, int64_t batches, int64_t owners)
{
    const int64_t slots = SegmentSlots(batches, owners);
    return queries > owners && maxQueryLen > 1 && slots <= MAX_SLOTS ? static_cast<uint32_t>(slots) : 0;
}

#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr int64_t
    AlignedOffset(int64_t bytes)
{
    return (bytes + SLOT_BYTES - 1) / SLOT_BYTES * SLOT_BYTES;
}
} // namespace SligBatchLoss

#endif
