/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QGMM_MX_UB_OUTPUT_POLICY_H
#define QGMM_MX_UB_OUTPUT_POLICY_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define QGMM_MX_POLICY_INLINE __aicore__ inline
#else
#define QGMM_MX_POLICY_INLINE inline
#endif

namespace QGmmMxUbPolicy {
// The two M/N windows cover the measured 256-aligned neighborhoods of
// (1792, 5120) and (5120, 3584). Larger shapes are not necessarily faster with UB.
constexpr int64_t SMALL_M_MIN = 1792;
constexpr int64_t SMALL_M_MAX = 2048;
constexpr int64_t LARGE_N_MIN = 4864;
constexpr int64_t LARGE_N_MAX = 5120;
constexpr int64_t LARGE_M_MIN = 4864;
constexpr int64_t LARGE_M_MAX = 5120;
constexpr int64_t SMALL_N_MIN = 3584;
constexpr int64_t SMALL_N_MAX = 3840;
// K_i samples at 2048, 8192 and 16384 showed no regression for six equal groups
// in these windows. Apply the same span to each nonempty group length.
constexpr int64_t MIN_GROUP_K = 2048;
constexpr int64_t MAX_GROUP_K = 16384;
constexpr int64_t TILE_DIM = 256;

struct GroupKPolicy {
    static QGMM_MX_POLICY_INLINE bool UseUb(int64_t k)
    {
        return k >= MIN_GROUP_K && k <= MAX_GROUP_K;
    }
};

QGMM_MX_POLICY_INLINE bool IsShapeCandidate(int64_t m, int64_t n, int64_t baseM, int64_t baseN, uint64_t groupCount,
                                            uint64_t coreCount)
{
    bool inNetworkFamily = (m >= SMALL_M_MIN && m <= SMALL_M_MAX && n >= LARGE_N_MIN && n <= LARGE_N_MAX) ||
                           (m >= LARGE_M_MIN && m <= LARGE_M_MAX && n >= SMALL_N_MIN && n <= SMALL_N_MAX);
    return inNetworkFamily && baseM == TILE_DIM && baseN == TILE_DIM && m % TILE_DIM == 0 && n % TILE_DIM == 0 &&
           groupCount > 0 && coreCount > 0;
}

QGMM_MX_POLICY_INLINE bool IsTotalKCandidate(int64_t totalK, uint64_t groupCount)
{
    if (totalK < MIN_GROUP_K || groupCount == 0) {
        return false;
    }
    // Host counts all groups, including empty ones. Apply the total-K bounds
    // without multiplying by the group count; the kernel still checks each group's K.
    const auto k = static_cast<uint64_t>(totalK);
    return k / groupCount >= MIN_GROUP_K && groupCount >= 1 + (k - 1) / MAX_GROUP_K;
}

} // namespace QGmmMxUbPolicy

#undef QGMM_MX_POLICY_INLINE
#endif
