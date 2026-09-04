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
 * \file test_indexer_quant_cache_golden.cpp
 * \brief Numeric golden tests for the IndexerQuantCache Normal quantization branch.
 */
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "data_utils.h"
#include "tikicpulib.h"
#endif

extern "C" __global__ __aicore__ void indexer_quant_cache(GM_ADDR cache, GM_ADDR cache_scale, GM_ADDR x,
                                                          GM_ADDR slot_mapping, GM_ADDR cache_out,
                                                          GM_ADDR cache_scale_out, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t BS = 4;
constexpr int64_t D = 64;
constexpr int64_t CACHE_ROWS = BS;
constexpr int64_t NORMAL_QUANT_MODE = 1;
constexpr int64_t TILING_KEY = 10001;
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;
constexpr std::array<int32_t, BS> SLOT_MAPPING = {0, -1, 2, 3};

struct NormalGolden {
    int64_t roundScale;
    std::array<uint8_t, 2> cache;
    uint32_t scale;
};

template <typename T>
T *GmAlloc(size_t size)
{
    T *ptr = reinterpret_cast<T *>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

void FillInput(uint8_t *input)
{
    uint16_t *halfInput = reinterpret_cast<uint16_t *>(input);
    for (int64_t i = 0; i < BS * D; ++i) {
        halfInput[i] = (i & 1) ? 0xB800 : 0x3C00; // -0.5, 1.0
    }
}

IndexerQuantCacheTilingData MakeTiling(const NormalGolden &golden)
{
    IndexerQuantCacheTilingData tiling;
    std::memset(&tiling, 0, sizeof(tiling));
    tiling.bs = BS;
    tiling.d = D;
    tiling.scaleCol = 1;
    tiling.rowOfFormerBlock = BS;
    tiling.rowOfTailBlock = BS;
    tiling.rowLoopOfFormerBlock = BS;
    tiling.rowLoopOfTailBlock = BS;
    tiling.rowFactor = 1;
    tiling.tailRowFactorOfFormerBlock = 1;
    tiling.tailRowFactorOfTailBlock = 1;
    tiling.quantMode = NORMAL_QUANT_MODE;
    tiling.roundScale = golden.roundScale;
    tiling.scalesAttr = 1.0f;
    tiling.blockSize = 1;
    tiling.cacheRowStride = D;
    tiling.cacheBlockStride = D;
    tiling.scaleRowStride = 1;
    tiling.scaleBlockStride = 1;
    return tiling;
}

void RunAndCheckNormalGolden(const NormalGolden &golden)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    constexpr size_t X_SIZE = BS * D * sizeof(uint16_t);
    constexpr size_t SLOT_SIZE = BS * sizeof(int32_t);
    constexpr size_t CACHE_SIZE = CACHE_ROWS * D * sizeof(uint8_t);
    constexpr size_t SCALE_SIZE = CACHE_ROWS * sizeof(uint32_t);

    uint8_t *cache = GmAlloc<uint8_t>(CACHE_SIZE);
    uint8_t *cacheScale = GmAlloc<uint8_t>(SCALE_SIZE);
    uint8_t *x = GmAlloc<uint8_t>(X_SIZE);
    uint8_t *slotMapping = GmAlloc<uint8_t>(SLOT_SIZE);
    uint8_t *cacheOut = GmAlloc<uint8_t>(CACHE_SIZE);
    uint8_t *cacheScaleOut = GmAlloc<uint8_t>(SCALE_SIZE);
    uint8_t *workspace = GmAlloc<uint8_t>(SYS_WORKSPACE);
    uint8_t *tiling = GmAlloc<uint8_t>(sizeof(IndexerQuantCacheTilingData));

    std::memset(cache, 0, CACHE_SIZE);
    std::memset(cacheScale, 0, SCALE_SIZE);
    FillInput(x);
    std::memcpy(slotMapping, SLOT_MAPPING.data(), SLOT_SIZE);
    std::memset(cacheOut, 0, CACHE_SIZE);
    std::memset(cacheScaleOut, 0, SCALE_SIZE);
    std::memset(workspace, 0, SYS_WORKSPACE);
    const IndexerQuantCacheTilingData tilingData = MakeTiling(golden);
    std::memcpy(tiling, &tilingData, sizeof(tilingData));

    ICPU_SET_TILING_KEY(TILING_KEY);
    ICPU_RUN_KF(indexer_quant_cache, 1, cache, cacheScale, x, slotMapping, cacheOut, cacheScaleOut, workspace, tiling);

    int64_t cacheMismatch = 0;
    int64_t scaleMismatch = 0;
    for (int64_t row = 0; row < BS; ++row) {
        const bool skipped = SLOT_MAPPING[row] < 0;
        for (int64_t col = 0; col < D; ++col) {
            const uint8_t expected = skipped ? 0 : golden.cache[col & 1];
            cacheMismatch += cache[row * D + col] != expected;
        }
        uint32_t actualScale = 0;
        std::memcpy(&actualScale, cacheScale + row * sizeof(uint32_t), sizeof(actualScale));
        scaleMismatch += actualScale != (skipped ? 0 : golden.scale);
    }
    EXPECT_EQ(cacheMismatch, 0) << "cache differs from the literal float8_e4m3fn golden";
    EXPECT_EQ(scaleMismatch, 0) << "cache scale differs from the literal float32 golden";

    AscendC::GmFree(cache);
    AscendC::GmFree(cacheScale);
    AscendC::GmFree(x);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(cacheOut);
    AscendC::GmFree(cacheScaleOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST(IndexerQuantCacheNormalGolden, RoundScale)
{
    // Input alternates 1.0/-0.5. 2^-8 scale produces E4M3FN bytes 0x78/0xF0.
    RunAndCheckNormalGolden(NormalGolden{1, {0x78, 0xF0}, 0x3B800000U});
}

TEST(IndexerQuantCacheNormalGolden, NoRoundScale)
{
    // Input alternates 1.0/-0.5. 1/448 scale produces E4M3FN bytes 0x7E/0xF6.
    RunAndCheckNormalGolden(NormalGolden{0, {0x7E, 0xF6}, 0x3B124925U});
}
