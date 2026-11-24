// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>
#include <cstddef>

// TILING_KEY_IS requires keys visible to the kernel preprocessor.
#define DLGMM_FP16_TILING_KEY 0
#define DLGMM_FP16_TRANSPOSE_TILING_KEY 1
#define DLGMM_BF16_TILING_KEY 2
#define DLGMM_BF16_TRANSPOSE_TILING_KEY 3

namespace DualLevelGroupedMatmul {
inline constexpr int64_t DEFAULT_LEVEL0_GROUP_SIZE = 512;
inline constexpr int64_t DEFAULT_LEVEL1_GROUP_SIZE = 32;
inline constexpr uint32_t LEVEL1_SCALE_L1_WINDOW_K = 4096;
inline constexpr int64_t MX_SCALE_PAIR = 2;
inline constexpr int64_t WEIGHT_NK_AXIS = 2;
inline constexpr uint64_t TILING_DATA_VERSION = 1;
inline constexpr uint64_t DEFAULT_BASE_M = 32;
inline constexpr uint64_t DEFAULT_BASE_N = 64;
inline constexpr size_t TILING_DATA_ALIGNMENT = 8;
inline constexpr size_t TILING_DATA_WORDS = 14;
inline constexpr size_t LEVEL0_SIZE_WORD = 8;
inline constexpr size_t OUTPUT_BF16_WORD = 13;
// Operator-owned Host/Kernel tiling data. Dimensions count logical elements.

struct alignas(TILING_DATA_ALIGNMENT) DualLevelGroupedMatmulTilingData {
    uint64_t version = TILING_DATA_VERSION;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t experts = 0;
    // Protocol v1 metadata; device addressing is derived from the logical shape.
    uint64_t weightStrideBytes = 0;
    // M/N for scheduling and L1 planning. BlockMmad defines its L0 limits.
    uint64_t baseM = DEFAULT_BASE_M;
    uint64_t baseN = DEFAULT_BASE_N;
    // Number of K elements per level0 scale group.
    uint64_t level0Size = DEFAULT_LEVEL0_GROUP_SIZE;
    uint64_t coreNum = 0;
    uint64_t transposeWeight = 0;
    uint64_t groupListType = 0;
    uint64_t hasBias = 0;
    uint64_t outputBf16 = 0;
};

static_assert(alignof(DualLevelGroupedMatmulTilingData) == TILING_DATA_ALIGNMENT);
static_assert(sizeof(DualLevelGroupedMatmulTilingData) == TILING_DATA_WORDS * sizeof(uint64_t));
static_assert(offsetof(DualLevelGroupedMatmulTilingData, level0Size) == LEVEL0_SIZE_WORD * sizeof(uint64_t));
static_assert(offsetof(DualLevelGroupedMatmulTilingData, outputBf16) == OUTPUT_BF16_WORD * sizeof(uint64_t));
} // namespace DualLevelGroupedMatmul
