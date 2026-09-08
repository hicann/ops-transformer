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
 * \file allto_allv_grouped_mat_mul_aiv_plan.h
 * \brief Host validation helpers for the AIV prefix tiling ABI.
 */
#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_PLAN_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_PLAN_H

#include <cstdint>
#include <limits>

#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"

namespace AlltoAllvGroupedMatMulAivPlan {
constexpr uint32_t kMaxCountNum = A2AVGMM_MAX_COUNT_NUM;
constexpr uint32_t kMaxLocalExpertNum = A2AVGMM_MAX_LOCAL_EXPERT_NUM;

inline bool SafeAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline bool SafeMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool IsValidCountShape(uint32_t rankSize, uint32_t expertPerRank, uint32_t &countNum)
{
    if (rankSize == 0U || expertPerRank == 0U || expertPerRank > kMaxLocalExpertNum) {
        return false;
    }
    uint64_t product = 0U;
    if (!SafeMulU64(rankSize, expertPerRank, product)) {
        return false;
    }
    if (product > kMaxCountNum) {
        return false;
    }
    countNum = static_cast<uint32_t>(product);
    return true;
}

inline bool BuildInclusivePrefixes(const int64_t *sendCounts, const int64_t *recvCounts, uint32_t rankSize,
                                   uint32_t expertPerRank, uint32_t inputM, uint32_t outputM, int32_t *sendPrefix,
                                   int32_t *recvPrefix)
{
    uint32_t countNum = 0U;
    if (sendCounts == nullptr || recvCounts == nullptr || sendPrefix == nullptr || recvPrefix == nullptr ||
        inputM == 0U || outputM == 0U || !IsValidCountShape(rankSize, expertPerRank, countNum)) {
        return false;
    }

    uint64_t sendTotal = 0U;
    for (uint32_t index = 0U; index < countNum; ++index) {
        const int64_t count = sendCounts[index];
        if (count < 0 || static_cast<uint64_t>(count) > inputM ||
            !SafeAddU64(sendTotal, static_cast<uint64_t>(count), sendTotal) ||
            sendTotal > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        sendPrefix[index] = static_cast<int32_t>(sendTotal);
    }

    uint64_t recvTotal = 0U;
    for (uint32_t expert = 0U; expert < expertPerRank; ++expert) {
        for (uint32_t source = 0U; source < rankSize; ++source) {
            const uint32_t inputIndex = source * expertPerRank + expert;
            const int64_t count = recvCounts[inputIndex];
            if (count < 0 || static_cast<uint64_t>(count) > outputM ||
                !SafeAddU64(recvTotal, static_cast<uint64_t>(count), recvTotal) ||
                recvTotal > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                return false;
            }
            recvPrefix[expert * rankSize + source] = static_cast<int32_t>(recvTotal);
        }
    }

    return sendTotal == inputM && recvTotal == outputM;
}
} // namespace AlltoAllvGroupedMatMulAivPlan

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_PLAN_H
