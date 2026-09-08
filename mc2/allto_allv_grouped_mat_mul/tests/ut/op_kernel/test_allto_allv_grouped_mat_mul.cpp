/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "allto_allv_grouped_mat_mul_hccl_context_stub.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h"

namespace {

namespace AivComm = AlltoAllvGroupedMatMulAiv;
namespace AivMode = AlltoAllvGroupedMatMulAivMode;
namespace AivCatlass = AlltoAllvGroupedMatMulCatlass;

TEST(AlltoAllvGroupedMatMulV2KernelTest, BuildsExpertSequenceOneAtATime)
{
    constexpr uint32_t rankSize = 2U;
    constexpr uint32_t expertPerRank = 4U;
    const int32_t recvPrefix[] = {3, 5, 5, 9, 14, 14, 15, 21};
    const std::array<uint64_t, expertPerRank> expectedBase = {0U, 5U, 9U, 14U};
    const std::array<uint32_t, expertPerRank> expectedCount = {5U, 4U, 5U, 7U};

    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        AivComm::ExpertMeta expert = {};
        ASSERT_TRUE(AivMode::BuildExpertMetaForIndex(recvPrefix, rankSize, expertPerRank, expertIdx, 21U, expert));
        EXPECT_EQ(expert.recvTokenBase, expectedBase[expertIdx]);
        EXPECT_EQ(expert.tokenCount, expectedCount[expertIdx]);
    }
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, ClassifiesAutomaticExpertOverlapSafety)
{
    constexpr uint32_t rankSize = 4U;
    constexpr uint32_t expertPerRank = 2U;
    const int32_t smallBalanced[] = {2, 4, 6, 8, 10, 12, 14, 16};
    const int32_t belowThreshold[] = {8, 16, 24, 32, 40, 48, 56, 63};
    const int32_t thresholdBalanced[] = {8, 16, 24, 32, 40, 48, 56, 64};
    const int32_t oneNonEmpty[] = {32, 64, 96, 128, 128, 128, 128, 128};
    const int32_t invalid[] = {16, 32, 48, 47, 64, 80, 96, 112};

    AivMode::ExpertOverlapStats stats = {};
    ASSERT_TRUE(AivMode::BuildExpertOverlapStats(smallBalanced, rankSize, expertPerRank, stats));
    EXPECT_EQ(stats.nonEmptyExperts, 2U);
    EXPECT_EQ(stats.cappedTotalTokens, 16U);
    EXPECT_TRUE(AivMode::IsExpertOverlapProtocolSafe(smallBalanced, rankSize, expertPerRank));
    EXPECT_FALSE(AivMode::ShouldUseAutomaticExpertOverlap(smallBalanced, rankSize, expertPerRank, 256U, 256U));

    EXPECT_FALSE(AivMode::ShouldUseAutomaticExpertOverlap(belowThreshold, rankSize, expertPerRank, 64U, 64U));
    EXPECT_TRUE(AivMode::ShouldUseAutomaticExpertOverlap(thresholdBalanced, rankSize, expertPerRank, 64U, 64U));
    EXPECT_FALSE(AivMode::ShouldUseAutomaticExpertOverlap(thresholdBalanced, rankSize, expertPerRank, 63U, 64U));
    EXPECT_FALSE(AivMode::ShouldUseAutomaticExpertOverlap(thresholdBalanced, rankSize, expertPerRank, 64U, 63U));
    EXPECT_FALSE(AivMode::IsExpertOverlapProtocolSafe(oneNonEmpty, rankSize, expertPerRank));
    EXPECT_FALSE(AivMode::BuildExpertOverlapStats(invalid, rankSize, expertPerRank, stats));
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, EnablesPerExpertOverlapOutsideLegacyExpertDomain)
{
    constexpr uint32_t rankSize = 2U;
    constexpr uint32_t expertPerRank = 129U;
    std::array<int32_t, rankSize * expertPerRank> recvPrefix = {};
    for (uint32_t index = 0U; index < recvPrefix.size(); ++index) {
        recvPrefix[index] = static_cast<int32_t>(index + 1U);
    }

    EXPECT_TRUE(AivMode::ShouldUseAutomaticExpertOverlap(recvPrefix.data(), rankSize, expertPerRank, 64U, 64U));
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, PreservesExpertMajorSourceMinorOffsets)
{
    constexpr uint32_t rankSize = 2U;
    constexpr uint32_t expertPerRank = 4U;
    const int32_t recvPrefix[] = {3, 5, 5, 9, 14, 14, 15, 21};
    const std::array<uint64_t, expertPerRank> expectedSourceOneOffset = {
        3U,
        5U,
        14U,
        15U,
    };

    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        uint64_t tokenOffset = 0U;
        ASSERT_TRUE(AivMode::GetDestinationSourceTokenOffset(recvPrefix, rankSize, expertPerRank, expertIdx, 1U, 21U,
                                                             tokenOffset));
        EXPECT_EQ(tokenOffset, expectedSourceOneOffset[expertIdx]);
    }
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, BuildsTransposedTailExpertGemm)
{
    AivComm::ExpertMeta expert = {};
    expert.recvTokenBase = 9U;
    expert.tokenCount = 5U;

    AivCatlass::GemmLaunchSpec spec = {};
    ASSERT_TRUE(AivCatlass::BuildExpertGemmSpec<true>(2U, expert, 272U, 130U, spec));
    EXPECT_EQ(spec.offsetA, 2448U);
    EXPECT_EQ(spec.offsetB, 70720U);
    EXPECT_EQ(spec.offsetC, 1170U);
    EXPECT_EQ(spec.lda, 272U);
    EXPECT_EQ(spec.ldb, 272U);
    EXPECT_EQ(spec.ldc, 130U);
    EXPECT_TRUE(spec.transposeB);
    EXPECT_TRUE(spec.hasWork);
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, SkipsZeroTokenExpertCompute)
{
    AivComm::ExpertMeta expert = {};
    expert.recvTokenBase = 5U;
    expert.tokenCount = 0U;

    AivCatlass::GemmLaunchSpec spec = {};
    ASSERT_TRUE(AivCatlass::BuildExpertGemmSpec<false>(1U, expert, 256U, 128U, spec));
    EXPECT_EQ(spec.offsetA, 1280U);
    EXPECT_EQ(spec.offsetB, 32768U);
    EXPECT_EQ(spec.offsetC, 640U);
    EXPECT_FALSE(spec.hasWork);
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, BuildsOptionalSharedExpertGemm)
{
    AivCatlass::GemmLaunchSpec spec = {};
    ASSERT_TRUE(AivCatlass::BuildSharedGemmSpec<false>(true, 16U, 256U, 128U, spec));
    EXPECT_EQ(spec.lda, 256U);
    EXPECT_EQ(spec.ldb, 128U);
    EXPECT_EQ(spec.ldc, 128U);
    EXPECT_TRUE(spec.hasWork);

    ASSERT_TRUE(AivCatlass::BuildSharedGemmSpec<true>(true, 16U, 256U, 128U, spec));
    EXPECT_EQ(spec.ldb, 256U);
    EXPECT_TRUE(spec.transposeB);

    EXPECT_FALSE(AivCatlass::BuildSharedGemmSpec<false>(false, 16U, 256U, 128U, spec));
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, NormalizesA2AndA3PeerContextMetadata)
{
    AivComm::PeerContextMetadata metadata = {};

    ASSERT_TRUE(AivComm::NormalizePeerContextMetadata(1U, 4U, 0U, metadata));
    EXPECT_EQ(metadata.rankId, 1U);
    EXPECT_EQ(metadata.rankSize, 4U);
    EXPECT_EQ(metadata.windowBytes, AivComm::kDefaultWindowBytes);

    ASSERT_TRUE(AivComm::NormalizePeerContextMetadata(2U, 8U, 64U * 1024U * 1024U, metadata));
    EXPECT_EQ(metadata.rankId, 2U);
    EXPECT_EQ(metadata.rankSize, 8U);
    EXPECT_EQ(metadata.windowBytes, 64U * 1024U * 1024U);
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, RejectsInvalidPeerContextMetadata)
{
    AivComm::PeerContextMetadata metadata = {};
    EXPECT_FALSE(AivComm::NormalizePeerContextMetadata(0U, 0U, 1U, metadata));
    EXPECT_FALSE(AivComm::NormalizePeerContextMetadata(0U, AivComm::kMaxRankSize + 1U, 1U, metadata));
    EXPECT_FALSE(AivComm::NormalizePeerContextMetadata(4U, 4U, 1U, metadata));
}

TEST(AlltoAllvGroupedMatMulV2KernelTest, AppliesArchitectureSpecificPeerRankLimits)
{
    for (const uint32_t rankSize : {2U, 4U, 8U}) {
        EXPECT_TRUE(AivComm::IsSupportedPeerRankSize(rankSize, false));
        EXPECT_TRUE(AivComm::IsSupportedPeerRankSize(rankSize, true));
    }
    for (const uint32_t rankSize : {16U, 32U, 64U, 128U}) {
        EXPECT_FALSE(AivComm::IsSupportedPeerRankSize(rankSize, false));
        EXPECT_TRUE(AivComm::IsSupportedPeerRankSize(rankSize, true));
    }
    for (const uint32_t rankSize : {0U, 1U, 3U, 9U, 129U}) {
        EXPECT_FALSE(AivComm::IsSupportedPeerRankSize(rankSize, false));
        EXPECT_FALSE(AivComm::IsSupportedPeerRankSize(rankSize, true));
    }
}

} // namespace
