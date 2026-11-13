/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Exercise the device sparse iterator on the host; only GM tensor reads need an adapter.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <gtest/gtest.h>

namespace {

#define __aicore__
namespace AscendC {
template <class T>
struct GlobalTensor {
    const std::vector<T>* data;
    T GetValue(uint64_t index) const
    {
        return data->at(index);
    }
};
} // namespace AscendC
#include "../../../op_kernel/arch22/gbsa_full_quant_utils.hpp"
#undef __aicore__
using namespace GbsaKernelArch22::Int8;

template <bool isPaged, bool hasResidualBlock = false>
void CheckSparseGeometry(uint32_t blockY, uint32_t visibleTokens, const std::vector<int32_t>& indices,
                         const std::vector<int32_t>& table, uint32_t headDim = 256)
{
    SparseSource<isPaged, hasResidualBlock, AscendC::GlobalTensor<int32_t>> source(
        {&indices}, {&table}, blockY, visibleTokens, table.size(), indices.size(), headDim);
    auto selected = indices;
    if (hasResidualBlock && visibleTokens % blockY != 0) {
        selected.push_back(visibleTokens / blockY);
    }
    std::vector<uint64_t> expected;
    uint32_t validBlocks = 0;
    for (int32_t id : selected) {
        const uint64_t start = uint64_t(id) * blockY;
        if (id < 0 || start >= visibleTokens || (isPaged && start / 128 >= table.size())) {
            continue;
        }
        ++validBlocks;
        for (uint64_t token = start; token < std::min(start + blockY, uint64_t(visibleTokens)); ++token) {
            expected.push_back(isPaged ? uint64_t(table.at(token / 128)) * 128 + token % 128 : token);
        }
    }
    const uint32_t rounds = (validBlocks + 512 / blockY - 1) / (512 / blockY);
    ASSERT_EQ(source.CountValid(indices.size()), validBlocks);
    uint32_t cursor = 0, cubeCursor = 0;
    std::vector<uint64_t> actual;
    for (uint32_t round = 0; round < rounds; ++round) {
        CopyPlan plan;
        auto tile = source.NextTile(cursor, validBlocks, &plan);
        auto cubeTile = source.DescribeTile(cubeCursor, validBlocks);
        KvCopyPlan copyPlan;
        constexpr uint32_t kvHeads = 3, kvHead = 1;
        const uint64_t kvStart = 37, kStride = 128 * kvHeads * headDim + 512;
        const uint64_t vStride = 128 * kvHeads * headDim + 1024;
        uint32_t recordCount = 0;
        for (uint32_t chunk = 0; chunk < 4; ++chunk) {
            ASSERT_EQ(cubeTile.pages[chunk].validTokens, tile.pages[chunk].validTokens);
            copyPlan.begin[chunk] = recordCount;
            source.ForEachKvRecord(cubeTile.pages[chunk], kStride, vStride, kvHeads, kvHead, kvStart,
                                   [&](const KvCopyRecord& record) { copyPlan.records[recordCount++] = record; });
            copyPlan.end[chunk] = recordCount;
        }
        uint32_t chunkId = 0;
        for (const auto& chunk : tile.pages) {
            std::vector<uint64_t> planned;
            ASSERT_TRUE(chunk.planBegin <= chunk.planEnd && chunk.planEnd <= FullQuantLayout::KV_TILE);
            for (uint32_t recordIndex = chunk.planBegin; recordIndex < chunk.planEnd; ++recordIndex) {
                const auto& span = plan.spans[recordIndex];
                for (uint32_t i = 0; i < span.length; ++i) {
                    planned.push_back(span.token + i);
                }
            }
            actual.insert(actual.end(), planned.begin(), planned.end());
            ASSERT_TRUE(planned.size() == chunk.validTokens && planned.size() <= 128);
            std::vector<uint64_t> keyAddresses, valueAddresses;
            for (uint32_t recordIndex = copyPlan.begin[chunkId]; recordIndex < copyPlan.end[chunkId]; ++recordIndex) {
                const auto& record = copyPlan.records[recordIndex];
                ASSERT_EQ(record.dstRow, keyAddresses.size());
                ASSERT_TRUE(record.ndNum == 1 || record.ndNum == 2);
                if (record.ndNum == 2) {
                    ASSERT_TRUE(record.keyStride && record.valueStride && record.keyStride % 32 == 0 &&
                                record.valueStride % 32 == 0);
                }
                for (uint32_t nd = 0; nd < record.ndNum; ++nd) {
                    for (uint32_t row = 0; row < record.rows; ++row) {
                        keyAddresses.push_back(record.key + nd * record.keyStride + row * kvHeads * headDim);
                        valueAddresses.push_back(record.value + nd * record.valueStride + row * kvHeads * headDim);
                    }
                }
            }
            ASSERT_EQ(keyAddresses.size(), planned.size());
            for (uint32_t i = 0; i < planned.size(); ++i) {
                const uint64_t token = planned[i];
                const uint64_t keyAddress = isPaged ?
                                                token / 128 * kStride + (token % 128 * kvHeads + kvHead) * headDim :
                                                ((kvStart + token) * kvHeads + kvHead) * headDim;
                const uint64_t valueAddress =
                    isPaged ? token / 128 * vStride + (token % 128 * kvHeads + kvHead) * headDim : keyAddress;
                ASSERT_EQ(keyAddresses[i], keyAddress);
                ASSERT_EQ(valueAddresses[i], valueAddress);
            }
            ++chunkId;
        }
    }
    ASSERT_EQ(actual, expected);
}

TEST(generic_block_sparse_attention_sparse_ut, int8_copy_windows)
{
    for (uint32_t headDim = 1; headDim <= 256; ++headDim) {
        for (uint32_t kvHeads = 1; kvHeads <= 16; ++kvHeads) {
            for (uint32_t kvHead = 0; kvHead < kvHeads; ++kvHead) {
                const auto window = MakeKvCopyWindow(headDim, kvHeads, kvHead);
                const uint32_t start = kvHead * headDim;
                ASSERT_TRUE(window.headOffset % 32U == 0);
                ASSERT_TRUE(window.headOffset <= start);
                ASSERT_TRUE(window.channels >= window.headOffset + headDim);
                ASSERT_TRUE(start - window.headOffset + window.channels <= kvHeads * headDim);
                ASSERT_TRUE(window.channels <= std::max(headDim, 64U) && window.channels <= 2 * headDim);
                ASSERT_TRUE(window.headOffset + (headDim + 31) / 32 * 32 <= (window.channels + 31) / 32 * 32);
                for (uint32_t d = 0; d < headDim; ++d) {
                    const uint32_t copied = start - window.headOffset + window.headOffset + d;
                    ASSERT_TRUE(copied / headDim == kvHead && copied % headDim == d);
                }
                if (kvHeads == 1) {
                    ASSERT_TRUE(window.channels == headDim && window.headOffset == 0);
                }
            }
        }
    }
}

TEST(generic_block_sparse_attention_sparse_ut, int8_sparse_copy_plans)
{
    std::mt19937 random(20261030);
    uint32_t cases = 0;
    for (uint32_t blockY : {1, 2, 4, 8, 16, 32, 64, 128}) {
        for (uint32_t visibleTokens : {0, 1, 7, 127, 128, 129, 255, 257, 511, 512, 513, 1537, 4097}) {
            std::vector<int32_t> table((visibleTokens + 127) / 128);
            std::iota(table.begin(), table.end(), 0);
            std::shuffle(table.begin(), table.end(), random);
            for (uint32_t pattern = 0; pattern < 5; ++pattern) {
                std::vector<int32_t> blockIds;
                for (uint32_t i = 0; i < (visibleTokens + blockY - 1) / blockY; ++i) {
                    if (pattern == 0 || (pattern == 1 && i % 2 == 0) || (pattern >= 2 && random() % 3 != 0)) {
                        blockIds.push_back(i);
                    }
                }
                // Public sparse indices must be sorted, unique and nonnegative.
                // Include sorted entries beyond the visibleTokens boundary to exercise clipping.
                if (pattern == 3 || pattern == 4) {
                    blockIds.push_back(1000000);
                }
                CheckSparseGeometry<false>(blockY, visibleTokens, blockIds, table);
                CheckSparseGeometry<true>(blockY, visibleTokens, blockIds, table);
                cases += 2;
                blockIds.erase(std::remove_if(blockIds.begin(), blockIds.end(),
                                              [&](int32_t id) { return uint64_t(id + 1) * blockY > visibleTokens; }),
                               blockIds.end());
                if (pattern == 4) {
                    blockIds.clear();
                }
                CheckSparseGeometry<false, true>(blockY, visibleTokens, blockIds, table);
                CheckSparseGeometry<true, true>(blockY, visibleTokens, blockIds, table);
                cases += 2;
                for (uint32_t headDim : {1, 17, 31, 32, 33, 64, 96, 127, 128, 129, 192, 250, 255}) {
                    CheckSparseGeometry<false, true>(blockY, visibleTokens, blockIds, table, headDim);
                    CheckSparseGeometry<true, true>(blockY, visibleTokens, blockIds, table, headDim);
                    CheckSparseGeometry<false, false>(blockY, visibleTokens, blockIds, table, headDim);
                    CheckSparseGeometry<true, false>(blockY, visibleTokens, blockIds, table, headDim);
                    cases += 4;
                }
            }
        }
    }
    EXPECT_EQ(cases, 29120U);
}

} // namespace
