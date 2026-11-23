/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include "../../mx_a8w4_pipeline_common.h"

#ifndef GMM_MX_BUFFERED_INLINE
#define GMM_MX_BUFFERED_INLINE constexpr
#define GMM_MX_BUFFERED_INLINE_LOCAL
#endif

namespace GROUPED_MATMUL {
namespace MxBuffered {
enum class Route : uint8_t {
    Original,
    WeightQuad = 4,
    OperandTriple = 3
};

GMM_MX_BUFFERED_INLINE bool HasSteadyStateWork(uint64_t k, Route route)
{
    const uint64_t stages = route == Route::WeightQuad ? WEIGHT_PIPELINE_STAGES : OPERAND_PIPELINE_STAGES;
    return k / K_TILE > stages;
}

GMM_MX_BUFFERED_INLINE Route Select(uint64_t groups, uint64_t k, uint64_t n, uint64_t cores, uint64_t baseM,
                                    bool hasBias, int64_t groupType, int64_t groupListType, bool singleX,
                                    bool singleWeight, bool singleY, uint64_t nBlocks)
{
    if (groups == 0 || cores != CUBE_CORES || baseM != MAX_GROUP_ROWS || hasBias || groupType != 0 ||
        groupListType != 1 || !singleX || !singleWeight || !singleY || k == 0 || k % K_TILE != 0 || n == 0 ||
        n % N_ALIGNMENT != 0 || nBlocks == 0) {
        return Route::Original;
    }
    // One narrow N wave within a scale window does not amortize quad fill/drain.
    if (nBlocks == cores && n <= FULL_M_ACCUMULATOR_N * nBlocks && k <= MX_SCALE_K_WINDOW) {
        return Route::Original;
    }
    // Candidate scheduling policy: one N wave can occupy the complete cube grid.
    // This is a performance heuristic, not a buffer-safety condition.
    const Route candidate = nBlocks >= cores ? Route::WeightQuad : Route::OperandTriple;
    // Filling and draining all slots without steady-state work does not amortize this pipeline.
    return HasSteadyStateWork(k, candidate) ? candidate : Route::Original;
}

template <class CountPtr>
GMM_MX_BUFFERED_INLINE bool FitsBufferedGroups(CountPtr counts, uint64_t groups, Route route, uint64_t maxN)
{
    uint64_t maxRows = route == Route::OperandTriple ? MAX_GROUP_ROWS - 1UL : MAX_GROUP_ROWS;
    if (route == Route::WeightQuad) {
        if (maxN == 0 || maxN > MAX_N_WIDTH) {
            return false;
        }
        // Round the accumulator row capacity down once instead of aligning every group's M.
        const uint64_t accumulatorRows = L0C_REGION_BYTES / (maxN * sizeof(float)) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
        maxRows = maxRows < accumulatorRows ? maxRows : accumulatorRows;
    }
    bool nonempty = false;
    for (uint64_t g = 0UL; g < groups; ++g) {
        const int64_t m = counts[g];
        if (m == 0) {
            continue;
        }
        if (m <= static_cast<int64_t>(REUSE_MIN_ROWS) || m > static_cast<int64_t>(maxRows)) {
            return false;
        }
        nonempty = true;
    }
    return nonempty;
}

GMM_MX_BUFFERED_INLINE bool MatchesNPartition(uint64_t mainCount, uint64_t mainSize, uint64_t firstCount,
                                              uint64_t firstSize, uint64_t secondCount, uint64_t secondSize,
                                              uint64_t totalN)
{
    // Each B stage is 64 KiB at K256. Equal N halves also need 32-column alignment.
    const bool mainValid =
        mainCount == 0 || (mainSize >= N_ALIGNMENT && mainSize <= MAX_N_WIDTH && mainSize % N_ALIGNMENT == 0);
    const bool firstValid =
        firstCount == 0 || (firstSize >= N_ALIGNMENT && firstSize <= MAX_N_WIDTH && firstSize % N_ALIGNMENT == 0);
    const bool secondValid =
        secondCount == 0 || (secondSize >= N_ALIGNMENT && secondSize <= MAX_N_WIDTH && secondSize % N_ALIGNMENT == 0);
    const uint64_t blocks = mainCount + firstCount + secondCount;
    const uint64_t coverage = mainCount * mainSize + firstCount * firstSize + secondCount * secondSize;
    const uint64_t lastSize = secondCount > 0 ? secondSize : (firstCount > 0 ? firstSize : mainSize);
    // The existing scheduler clamps the final block to the actual N size.
    return mainValid && firstValid && secondValid && blocks > 0 && blocks <= totalN / N_ALIGNMENT &&
           coverage >= totalN && coverage - lastSize < totalN;
}
} // namespace MxBuffered
} // namespace GROUPED_MATMUL

#ifdef GMM_MX_BUFFERED_INLINE_LOCAL
#undef GMM_MX_BUFFERED_INLINE
#undef GMM_MX_BUFFERED_INLINE_LOCAL
#endif
