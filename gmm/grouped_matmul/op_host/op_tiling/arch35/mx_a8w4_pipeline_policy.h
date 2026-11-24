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
#include "../../../op_kernel/mx_a8w4_pipeline_common.h"
namespace GROUPED_MATMUL {
namespace MxPipeline {
struct Eligibility {
    uint64_t originalKey;
    int64_t groupListType;
    int64_t groupType;
    uint64_t cores;
    bool hasBias;
    bool fp8Fp4;
    bool mxScale;
    bool bf16Output;
};
constexpr bool IsEligible(const Eligibility& p, uint64_t originalMxKey)
{
    return p.originalKey == originalMxKey && !p.hasBias && p.groupListType == 1 && p.groupType == 0 &&
           p.cores == ALIGNED_CORE_COUNT && p.fp8Fp4 && p.mxScale && p.bf16Output;
}
constexpr uint64_t MIN_OPTIMIZED_N = 2048UL;
constexpr uint64_t MIN_ALIGNED_CORE_WAVES = 2UL;
// Measured pipeline startup floor; shorter K retains the original implementation.
constexpr uint64_t MIN_ALIGNED_K = 2048UL;
constexpr uint64_t MIN_BUFFERED_AVERAGE_ROWS = MxBuffered::REUSE_MIN_ROWS + MxBuffered::CUBE_ALIGNMENT;
// Conservative measured performance limit for Buffered only; Aligned is selected independently above.
constexpr uint64_t MAX_BUFFERED_GROUPS = 128UL;
constexpr uint64_t SINGLE_BUFFERED_N_WAVE = ALIGNED_CORE_COUNT * MxBuffered::MAX_N_WIDTH;
constexpr Route Select(uint64_t m, uint64_t e, uint64_t k, uint64_t n)
{
    if (e == 0 || n < MIN_OPTIMIZED_N || m / e <= MxBuffered::REUSE_MIN_ROWS) {
        return Route::Original;
    }
    if (k >= MIN_ALIGNED_K && k % ALIGNED_TILE_SIZE == 0 && n % ALIGNED_TILE_SIZE == 0) {
        const uint64_t nTiles = n / ALIGNED_TILE_SIZE;
        const uint64_t rowsPerWave = (ALIGNED_CORE_COUNT + nTiles - 1) / nTiles;
        const uint64_t completeWaves = m / e / ALIGNED_TILE_SIZE / rowsPerWave;
        // Require enough full waves to amortize pipeline startup and draining.
        if (completeWaves >= MIN_ALIGNED_CORE_WAVES) {
            return Route::DynamicAligned;
        }
    }
    if ((n <= SINGLE_BUFFERED_N_WAVE || k >= MxBuffered::MX_SCALE_K_WINDOW) && e <= MAX_BUFFERED_GROUPS &&
        m / e > MIN_BUFFERED_AVERAGE_ROWS && m / e < MxBuffered::MAX_GROUP_ROWS && k % MxBuffered::K_TILE == 0 &&
        k / MxBuffered::K_TILE > MxBuffered::OPERAND_PIPELINE_STAGES && n % MxBuffered::N_ALIGNMENT == 0) {
        return Route::DynamicBuffered;
    }
    return Route::Original;
}
} // namespace MxPipeline
} // namespace GROUPED_MATMUL
