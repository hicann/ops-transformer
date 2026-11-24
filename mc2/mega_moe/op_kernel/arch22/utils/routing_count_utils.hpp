/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ROUTING_COUNT_UTILS_HPP
#define MEGA_MOE_ROUTING_COUNT_UTILS_HPP

#include <cstdint>
#include "catlass/catlass.hpp"

namespace Catlass::Gemm::Kernel {

// These helpers only update scalar counts. Synchronization and data movement
// stay in the caller at their original positions.
CATLASS_DEVICE void AccumulateSingleDequantRows(uint32_t currentM, uint64_t maxOutputSize, uint32_t& sum)
{
    if (sum + currentM <= maxOutputSize) {
        sum += currentM;
    } else if (sum < maxOutputSize) {
        sum = maxOutputSize;
    }
}

CATLASS_DEVICE void AccumulateSplitDequantRows(uint32_t currentM, int32_t groupIdx, uint32_t epilogueGranularity,
                                               uint64_t maxOutputSize, uint32_t& sum1, uint32_t& sum2)
{
    if (groupIdx + 1 <= epilogueGranularity) {
        if (sum1 + currentM <= maxOutputSize) {
            sum1 += currentM;
        } else if (sum1 < maxOutputSize) {
            sum1 = maxOutputSize;
        }
    }
    if (groupIdx + 1 > epilogueGranularity && sum1 < maxOutputSize) {
        if (sum1 + sum2 + currentM <= maxOutputSize) {
            sum2 += currentM;
        } else if (sum1 + sum2 < maxOutputSize) {
            sum2 += maxOutputSize - sum1 - sum2;
        }
    }
}

CATLASS_DEVICE bool IsSyncTask(int32_t task_id, int32_t n_tasks)
{
    int32_t offset = n_tasks - task_id;
    if (offset <= 0 || (offset & (offset - 1)) != 0) {
        return false;
    } else {
        return true;
    }
}

template <bool BatchedDispatch, class Params>
CATLASS_DEVICE bool UseDecodeFastPath(Params const& params)
{
    return BatchedDispatch && params.decodeFastPath;
}

template <bool BatchedDispatch, class Params>
CATLASS_DEVICE bool IsActivationBoundary(Params const& params, uint32_t groupIdx)
{
    if (UseDecodeFastPath<BatchedDispatch>(params)) {
        return groupIdx + 1 == params.epilogueGranularity || groupIdx + 1 == params.expertPerRank;
    }
    return IsSyncTask(groupIdx, params.expertPerRank);
}

} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_ROUTING_COUNT_UTILS_HPP
