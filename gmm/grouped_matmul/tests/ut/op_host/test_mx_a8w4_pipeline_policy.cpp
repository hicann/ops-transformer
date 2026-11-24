/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/mx_a8w4_pipeline_policy.h"
#include "../../../op_kernel/arch35/quant_adaptive_sliding_window_templates/gmm_weight_quant_mx_buffered_policy.h"

namespace {
namespace Host = GROUPED_MATMUL::MxPipeline;
namespace Device = GROUPED_MATMUL::MxBuffered;
constexpr uint64_t ORIGINAL_KEY = 74530817UL; // Frozen ABI expectation, not a production route constant.

TEST(GroupedMatmulMxPipelinePolicy, EligibilityAndCompactLayout)
{
    const Host::Eligibility valid{ORIGINAL_KEY, 1, 0, 32, false, true, true, true};
    EXPECT_TRUE(Host::IsEligible(valid, ORIGINAL_KEY));
    EXPECT_FALSE(Host::IsEligible(valid, ORIGINAL_KEY + 1));
    EXPECT_EQ(sizeof(Host::Eligibility), 5 * sizeof(uint64_t));
    EXPECT_GT(offsetof(Host::Eligibility, hasBias), offsetof(Host::Eligibility, cores));
    auto invalid = valid;
    invalid.hasBias = true;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.cores = 31;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.groupListType = 0;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.groupType = 1;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.fp8Fp4 = false;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.mxScale = false;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
    invalid = valid;
    invalid.bf16Output = false;
    EXPECT_FALSE(Host::IsEligible(invalid, ORIGINAL_KEY));
}

TEST(GroupedMatmulMxPipelinePolicy, AlignedWaveBoundaries)
{
    EXPECT_EQ(Host::Select(1024, 1, 1024, 2048), Host::Route::Original);
    EXPECT_EQ(Host::Select(1024, 1, 4096, 2048), Host::Route::Original);
    EXPECT_EQ(Host::Select(2048, 1, 1024, 2048), Host::Route::Original);
    EXPECT_EQ(Host::Select(2048, 1, 2048, 2048), Host::Route::DynamicAligned);
    EXPECT_EQ(Host::Select(32768, 4, 7168, 6144), Host::Route::DynamicAligned);
    EXPECT_EQ(Host::Select(32768, 4, 7167, 6144), Host::Route::Original);
    EXPECT_EQ(Host::Select(32768, 4, 7168, 6145), Host::Route::Original);
    EXPECT_EQ(Host::Select(32768, 0, 7168, 6144), Host::Route::Original);
    EXPECT_EQ(Host::Select(32768, 4, 7168, 1024), Host::Route::Original);
}

TEST(GroupedMatmulMxPipelinePolicy, BufferedRowsAndExpertLimit)
{
    for (uint64_t rows : {144UL, 145UL, 160UL, 255UL, 256UL}) {
        const auto expected = rows > 144 && rows < 256 ? Host::Route::DynamicBuffered : Host::Route::Original;
        EXPECT_EQ(Host::Select(rows, 1, 1024, 2112), expected);
    }
    EXPECT_EQ(Host::Select(128 * 160, 128, 3072, 2112), Host::Route::DynamicBuffered);
    EXPECT_EQ(Host::Select(129 * 160, 129, 3072, 2112), Host::Route::Original);
    EXPECT_EQ(Host::Select(129 * 2048, 129, 2048, 2048), Host::Route::DynamicAligned);
}

TEST(GroupedMatmulMxPipelinePolicy, ShortKAndWideNStartupBoundaries)
{
    for (uint64_t k : {1024UL, 1792UL, 1920UL, 2176UL}) {
        EXPECT_EQ(Host::Select(4096, 1, k, 8192), Host::Route::Original);
    }
    for (uint64_t k : {2048UL, 2304UL}) {
        EXPECT_EQ(Host::Select(4096, 1, k, 8192), Host::Route::DynamicAligned);
    }
    EXPECT_EQ(Host::Select(160, 1, 3072, 8192), Host::Route::DynamicBuffered);
    EXPECT_EQ(Host::Select(160, 1, 3072, 8256), Host::Route::Original);
    EXPECT_EQ(Host::Select(160, 1, 4096, 8256), Host::Route::DynamicBuffered);
}

TEST(GroupedMatmulMxPipelinePolicy, SteadyStateAndNPartition)
{
    EXPECT_FALSE(Device::HasSteadyStateWork(768, Device::Route::OperandTriple));
    EXPECT_TRUE(Device::HasSteadyStateWork(1024, Device::Route::OperandTriple));
    EXPECT_FALSE(Device::HasSteadyStateWork(1024, Device::Route::WeightQuad));
    EXPECT_TRUE(Device::HasSteadyStateWork(1280, Device::Route::WeightQuad));
    EXPECT_EQ(Device::Select(14, 1024, 6144, 32, 256, false, 0, 1, true, true, true, 32), Device::Route::Original);
    EXPECT_EQ(Device::Select(7, 1024, 8192, 32, 256, false, 0, 1, true, true, true, 64), Device::Route::Original);
    EXPECT_EQ(Device::Select(14, 3584, 6144, 32, 256, false, 0, 1, true, true, true, 32), Device::Route::WeightQuad);
    EXPECT_EQ(Device::Select(14, 3072, 3584, 32, 256, false, 0, 1, true, true, true, 16), Device::Route::OperandTriple);
    EXPECT_EQ(Device::Select(3, 1280, 8192, 32, 256, false, 0, 1, true, true, true, 32), Device::Route::WeightQuad);
    EXPECT_EQ(Device::Select(3, 4096, 4096, 32, 256, false, 0, 1, true, true, true, 32), Device::Route::Original);
    EXPECT_TRUE(Device::MatchesNPartition(0, 256, 32, 192, 0, 0, 6144));
    EXPECT_FALSE(Device::MatchesNPartition(0, 256, 32, 512, 0, 0, 6144));
    EXPECT_FALSE(Device::MatchesNPartition(0, 256, 31, 192, 0, 0, 6144));
}

TEST(GroupedMatmulMxPipelinePolicy, ExpertRowsAndAccumulatorCapacity)
{
    for (int64_t rows = -1; rows <= 257; ++rows) {
        for (uint64_t width = 32; width <= 256; width += 32) {
            const std::array<int64_t, 2> counts{0, rows};
            const bool triple = rows > 128 && rows < 256;
            const bool quad = rows > 128 && rows <= 256 &&
                              ((static_cast<uint64_t>(rows) + 15) / 16 * 16) * width * sizeof(float) <= 128 * 1024;
            EXPECT_EQ(Device::FitsBufferedGroups(counts.data(), counts.size(), Device::Route::OperandTriple, width),
                      triple);
            EXPECT_EQ(Device::FitsBufferedGroups(counts.data(), counts.size(), Device::Route::WeightQuad, width), quad);
        }
    }
    const std::array<int64_t, 2> empty{0, 0};
    EXPECT_FALSE(Device::FitsBufferedGroups(empty.data(), empty.size(), Device::Route::WeightQuad, 128));
    const std::array<int64_t, 2> counts{129, 0};
    EXPECT_FALSE(Device::FitsBufferedGroups(counts.data(), counts.size(), Device::Route::WeightQuad, 0));
    EXPECT_FALSE(Device::FitsBufferedGroups(counts.data(), counts.size(), Device::Route::WeightQuad, UINT64_MAX));
}
} // namespace
