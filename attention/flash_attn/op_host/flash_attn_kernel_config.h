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
 * \file flash_attn_kernel_config.h
 * \brief FA 分核/切块配置的计算接口
 */

#ifndef FLASH_ATTN_KERNEL_CONFIG_H
#define FLASH_ATTN_KERNEL_CONFIG_H

#include <cstdint>
#include "../op_kernel/utils/flash_attn_common_def.h"

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

inline uint32_t GetMaxUsedAicCores(uint32_t aicNum, uint32_t batchSize, uint32_t kvHeads, uint32_t groupSize,
                                   int64_t maxSeqQ, int64_t maxSeqKv, uint32_t mBaseSize, uint32_t s2BaseSize)
{
    if (aicNum == 0 || batchSize == 0 || kvHeads == 0 || groupSize == 0 || maxSeqQ <= 0 || maxSeqKv <= 0 ||
        mBaseSize == 0 || s2BaseSize == 0) {
        return aicNum;
    }
    // 分核方式是BN2_GS1_S2
    // Both tiling and metadata must use the same static bounds, without reading sequence tensors.
    // Count dense (B, N2, ceil(Q * G / M), ceil(KV / N)) blocks to retain split-KV parallelism.
    // Unknown/empty dimensions keep the platform limit, including the existing empty-output path.
    // Cap before multiplying Q by G, so even very large static bounds cannot overflow.
    if (static_cast<uint64_t>(maxSeqQ) > static_cast<uint64_t>(aicNum) * mBaseSize / groupSize) {
        return aicNum;
    }
    const uint64_t mSize = static_cast<uint64_t>(maxSeqQ) * groupSize;
    const uint64_t mBlocks = (mSize - 1) / mBaseSize + 1;
    const uint64_t s2Blocks = (static_cast<uint64_t>(maxSeqKv) - 1) / s2BaseSize + 1;
    uint32_t cores = 1;
    for (uint64_t factor : {static_cast<uint64_t>(batchSize), static_cast<uint64_t>(kvHeads), mBlocks, s2Blocks}) {
        if (factor >= (static_cast<uint64_t>(aicNum) - 1) / cores + 1) {
            return aicNum;
        }
        cores *= static_cast<uint32_t>(factor);
    }
    return cores;
}

inline uint32_t GetMaxUsedAicCores(uint32_t aicNum, uint32_t batchSize, uint32_t kvHeads, uint32_t groupSize,
                                   int64_t maxSeqQ, int64_t maxSeqKv, uint32_t mBaseSize, uint32_t s2BaseSize,
                                   uint32_t templateId)
{
    if (aicNum == 0 || batchSize == 0 || kvHeads == 0 || groupSize == 0 || maxSeqQ <= 0 || maxSeqKv <= 0 ||
        mBaseSize == 0 || s2BaseSize == 0) {
        return aicNum;
    }
    if (templateId == FA_Template_DN_Unmerged) {
        // 分核方式是BN1_S1_S2
        uint64_t mBlocks = (static_cast<uint64_t>(maxSeqQ) + mBaseSize - 1) / mBaseSize;
        uint64_t s2Blocks = (static_cast<uint64_t>(maxSeqKv) + s2BaseSize - 1) / s2BaseSize;
        uint64_t cores = static_cast<uint64_t>(batchSize) * kvHeads * groupSize * mBlocks * s2Blocks;
        return (cores < static_cast<uint64_t>(aicNum)) ? cores : aicNum;
    } else {
        return GetMaxUsedAicCores(aicNum, batchSize, kvHeads, groupSize, maxSeqQ, maxSeqKv, mBaseSize, s2BaseSize);
    }
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_KERNEL_CONFIG_H
