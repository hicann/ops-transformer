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
 * \file engram_fetch_utils.h
 * \brief engram_fetch算子公共头文件
 */

#ifndef ENGRAM_FETCH_UTILS_H
#define ENGRAM_FETCH_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Kernel {
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t TILE_BYTES = 32U * 1024U;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t READ_COMMIT_FIRST_THRESHOLD = 4U;
constexpr uint32_t READ_COMMIT_THRESHOLD = 16U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t RELAY_BUFFER_NUM = 2U;

constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t WIN_REGION_COUNT = 6U;
constexpr uint32_t NUM_SLOTS = 4U;
constexpr uint32_t INDICES_RATIO = 50U;
constexpr uint32_t UB_RESERVED_SIZE = 8U * 1024U;
constexpr uint32_t SENDER_CHANNEL_IDX = 0U;
constexpr uint32_t RECEIVER_CHANNEL_IDX = 1U;
constexpr uint32_t HANDLE_ARRAY_SIZE = 72U;

struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE * 2];
    uint32_t channelsPerRank;
};

struct CoreAssignment {
    uint32_t assignedRank;
    uint32_t idxInRankGroup;
    uint32_t rankGroupSize;
};

__aicore__ inline CoreAssignment GetCoreAssignment(uint32_t totalBlocks, uint32_t aivId, uint32_t numRanks)
{
    CoreAssignment result{numRanks, 0, 0};
    uint32_t base = totalBlocks / numRanks;
    uint32_t remainder = totalBlocks % numRanks;
    uint32_t accumulated = 0;
    for (uint32_t r = 0; r < numRanks; r++) {
        uint32_t groupSize = base + ((r < remainder) ? 1U : 0U);
        if (aivId < accumulated + groupSize) {
            result.assignedRank = r;
            result.rankGroupSize = groupSize;
            result.idxInRankGroup = aivId - accumulated;
            return result;
        }
        accumulated += groupSize;
    }
    return result;
}

} // namespace Mc2Kernel

#endif
