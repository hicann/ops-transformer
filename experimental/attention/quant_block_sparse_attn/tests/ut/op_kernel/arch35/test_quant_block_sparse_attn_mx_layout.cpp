/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_kernel/common/util_regbase_mx.h"

namespace QuantBlockSparseAttnUT {
namespace {
template <QBSALayout layout>
void CheckPaddedLayout()
{
    constexpr uint32_t batches = 3, heads = 8, sq = 129;
    std::vector<uint64_t> slots;
    // Independent storage order; each cell records its logical B,N,S identity.
    for (uint32_t b = 0; b < batches; ++b) {
        const uint32_t outer = layout == QBSALayout::BSND ? sq : heads;
        const uint32_t inner = layout == QBSALayout::BSND ? heads : sq;
        for (uint32_t i = 0; i < outer; ++i) {
            for (uint32_t j = 0; j < inner; ++j) {
                const uint32_t n = layout == QBSALayout::BSND ? j : i;
                const uint32_t s = layout == QBSALayout::BSND ? i : j;
                slots.push_back((b * heads + n) * sq + s);
            }
        }
    }
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t n = 0; n < heads; ++n) {
            for (uint32_t s = 0; s < sq; ++s) {
                const uint64_t expected = (b * heads + n) * sq + s;
                const auto slot = regbasemx::MxQuerySlot<layout>(b * sq, s, n, heads, sq);
                ASSERT_LT(slot, slots.size());
                EXPECT_EQ(slots[slot], expected);
                EXPECT_EQ(regbasemx::MxLseSlot<layout>(b * sq, s, n, heads, sq), expected);
            }
        }
    }
}

TEST(QbsaMxQueryLayout, BSND)
{
    CheckPaddedLayout<QBSALayout::BSND>();
}
TEST(QbsaMxQueryLayout, BNSD)
{
    CheckPaddedLayout<QBSALayout::BNSD>();
}
TEST(QbsaMxQueryLayout, PackedTndAndWideOffsets)
{
    const uint64_t base = uint64_t{1} << 30;
    EXPECT_EQ(regbasemx::MxQuerySlot<QBSALayout::TND>(base, 7, 3, 8, 0), (base + 7) * 8 + 3);
    EXPECT_EQ(regbasemx::MxLseSlot<QBSALayout::TND>(base, 7, 3, 8, 0), (base + 7) * 8 + 3);
    EXPECT_EQ(regbasemx::MxQuerySlot<QBSALayout::BNSD>(base, 7, 3, 8, 129), base * 8 + 3 * 129 + 7);
}

TEST(QbsaMxQueryLayout, D64AndD256UseLayoutSpecificRowOffsets)
{
    constexpr uint32_t batch = 1U;
    constexpr uint32_t heads = 8U;
    constexpr uint32_t sq = 129U;
    constexpr uint32_t head = 3U;
    constexpr uint32_t sequence = 17U;
    constexpr uint32_t rows = 32U;
    constexpr uint64_t tokenBase = batch * sq;
    for (const uint32_t headDim : {64U, 256U}) {
        const uint64_t bsndBegin =
            regbasemx::MxQuerySlot<QBSALayout::BSND>(tokenBase, sequence, head, heads, sq) * headDim;
        const uint64_t bsndEnd =
            regbasemx::MxQuerySlot<QBSALayout::BSND>(tokenBase, sequence + rows, head, heads, sq) * headDim;
        EXPECT_EQ(bsndEnd - bsndBegin, static_cast<uint64_t>(rows) * heads * headDim);

        const uint64_t bnsdBegin =
            regbasemx::MxQuerySlot<QBSALayout::BNSD>(tokenBase, sequence, head, heads, sq) * headDim;
        const uint64_t bnsdEnd =
            regbasemx::MxQuerySlot<QBSALayout::BNSD>(tokenBase, sequence + rows, head, heads, sq) * headDim;
        EXPECT_EQ(bnsdEnd - bnsdBegin, static_cast<uint64_t>(rows) * headDim);
    }
}

TEST(QbsaMxQueryLayout, MetadataEndsAreClippedToPhysicalQuery)
{
    EXPECT_EQ(regbasemx::MxS1LoopEnd(129, 64, 0), 3U);
    EXPECT_EQ(regbasemx::MxS1LoopEnd(129, 64, 2), 2U);
    EXPECT_EQ(regbasemx::MxS1LoopEnd(129, 64, 7), 3U);
    EXPECT_EQ(regbasemx::MxS1LoopEnd(129, 128, 7), 2U);
    EXPECT_EQ(regbasemx::MxS1LoopEnd(0, 128, 7), 0U);
    EXPECT_EQ(regbasemx::MxS1LoopEnd(UINT32_MAX, 128, 0), 33554432U);
}
} // namespace
} // namespace QuantBlockSparseAttnUT
