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

namespace GROUPED_MATMUL {
namespace MxBuffered {
// Resource limits must match the paired Blaze producer and consumer layouts.
constexpr uint64_t CUBE_CORES = 32UL;
constexpr uint64_t K_TILE = 256UL;
constexpr uint64_t B_STAGE_BYTES = 64UL * 1024UL;
constexpr uint64_t L0C_REGION_BYTES = 128UL * 1024UL;
constexpr uint64_t CUBE_ALIGNMENT = 16UL;
constexpr uint64_t N_ALIGNMENT = 32UL;
constexpr uint64_t MAX_GROUP_ROWS = B_STAGE_BYTES / K_TILE;
constexpr uint64_t MAX_N_WIDTH = B_STAGE_BYTES / K_TILE;
constexpr uint64_t FULL_M_ACCUMULATOR_N = L0C_REGION_BYTES / (MAX_GROUP_ROWS * sizeof(float));
constexpr uint64_t MX_SCALE_K_WINDOW = 4096UL;
constexpr uint64_t REUSE_MIN_ROWS = MAX_GROUP_ROWS / 2UL;
constexpr uint64_t OPERAND_PIPELINE_STAGES = 3UL;
constexpr uint64_t WEIGHT_PIPELINE_STAGES = 4UL;

} // namespace MxBuffered
namespace MxPipeline {
// Algorithm categories occupy bits [32:29]; keep category zero as the original route.
enum class Route : uint8_t {
    Original = 0,
    DynamicAligned = 3,
    DynamicBuffered = 4
};
constexpr uint64_t ALIGNED_CORE_COUNT = 32UL;
constexpr uint64_t ALIGNED_TILE_SIZE = 256UL;
} // namespace MxPipeline
} // namespace GROUPED_MATMUL
