/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "../../../op_kernel/catlass_hc_forward_mul_sum_tiling_data.h"
#include "../../../op_host/text_gated_residual_tiling.h"
#include "tiling_case_executor.h"

TEST(TextGatedResidualTiling, SmallAndLargeBothDtypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t m : {32, 1024}) {
            optiling::TextGatedResidualCompileInfo compileInfo{};
            gert::TilingContextPara para("TextGatedResidual",
                                         {{{{1, m, 2048}, {1, m, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{64, 2048}, {64, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{2048, 64}, {2048, 64}}, dtype, ge::FORMAT_ND},
                                          {{{4, 2048}, {4, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                         {{{{1, m, 512}, {1, m, 512}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 4}, {1, m, 4}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 4}, {1, m, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          {{{1, m, 64}, {1, m, 64}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 2048}, {1, m, 2048}}, dtype, ge::FORMAT_ND}},
                                         {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                          {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                          {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                         &compileInfo, "Ascend910B", 20, 196608, 65536);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            EXPECT_GT(info.blockNum, 0);
            EXPECT_GE(info.tilingDataSize, 128);
        }
    }
}

namespace {
bool RunSmallModelRouting(int64_t m, int64_t r, int64_t c, int64_t h, bool backward, uint64_t cores,
                          const std::string& soc, ge::DataType dtype, bool gate, bool gamma, TilingInfo& info)
{
    const int64_t d = c * h;
    optiling::TextGatedResidualCompileInfo compileInfo{};
    using Desc = gert::TilingContextPara::TensorDescription;
    std::vector<Desc> inputs = {{{{1, m, d}, {1, m, d}}, dtype, ge::FORMAT_ND},
                                {{{r, d}, {r, d}}, dtype, ge::FORMAT_ND},
                                {{{d, r}, {d, r}}, dtype, ge::FORMAT_ND}};
    if (gate)
        inputs.emplace_back(gert::StorageShape{{c, d}, {c, d}}, dtype, ge::FORMAT_ND);
    if (gamma)
        inputs.emplace_back(gert::StorageShape{{d}, {d}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("TextGatedResidual", inputs,
                                 {{{{1, m, h}, {1, m, h}}, dtype, ge::FORMAT_ND},
                                  {{{1, m, c}, {1, m, c}}, dtype, ge::FORMAT_ND},
                                  {{{1, m, c}, {1, m, c}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{1, m, r}, {1, m, r}}, dtype, ge::FORMAT_ND},
                                  {{{1, m, d}, {1, m, d}}, dtype, ge::FORMAT_ND}},
                                 {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c)},
                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                  {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(backward)}},
                                 {1, 1, 1, gate ? 1u : 0u, gamma ? 1u : 0u}, {1, 1, 1, 1, 1}, &compileInfo, soc, cores,
                                 262144, 4096);
    if (soc == "Ascend950") {
        para.socInfoString_ = R"({"hardware_info":{"UB_SIZE":262144,"L1_SIZE":524288,
            "L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144,"CORE_NUM":)" +
                              std::to_string(cores) + R"(,"socVersion":"Ascend950"}})";
    }
    return ExecuteTiling(para, info);
}
} // namespace

TEST(TextGatedResidualTiling, A1B4ExactDomainAndOptionalInputs)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t m : {1, 16, 32, 64, 128, 256}) {
            for (bool gate : {false, true}) {
                for (bool gamma : {false, true}) {
                    TilingInfo info;
                    ASSERT_TRUE(
                        RunSmallModelRouting(m, 320, 4, 2560, false, 28, "Ascend950", dtype, gate, gamma, info));
                    const uint32_t expectedKey = m <= 16 ? 14u : (m <= 64 ? 4u : (m == 128 ? 6u : 7u));
                    EXPECT_EQ(info.tilingKey, expectedKey);
                    EXPECT_EQ(info.blockNum, 28);
                    EXPECT_EQ(info.tilingDataSize, 128);
                }
            }
        }
    }
}

TEST(TextGatedResidualTiling, A1B4RoutingAndFallback)
{
    auto fallback = [](int64_t m, int64_t r, int64_t c, int64_t h, bool backward, uint64_t cores,
                       const std::string& soc, int64_t expectedKey = 3) {
        TilingInfo info;
        ASSERT_TRUE(RunSmallModelRouting(m, r, c, h, backward, cores, soc, ge::DT_BF16, true, true, info));
        EXPECT_EQ(info.tilingKey, expectedKey);
    };
    fallback(0, 320, 4, 2560, false, 28, "Ascend950");
    fallback(2, 320, 4, 2560, false, 28, "Ascend950", 14);
    fallback(127, 320, 4, 2560, false, 28, "Ascend950", 6);
    for (int64_t m : {129, 255, 257, 512})
        fallback(m, 320, 4, 2560, false, 28, "Ascend950", 7);
    for (int64_t r : {256, 384})
        fallback(16, r, 4, 2560, false, 28, "Ascend950");
    for (int64_t c : {2, 8})
        fallback(16, 320, c, 2560, false, 28, "Ascend950");
    for (int64_t h : {2048, 4096})
        fallback(16, 320, 4, h, false, 28, "Ascend950");
    fallback(16, 320, 4, 2560, true, 28, "Ascend950", 21);
    for (uint64_t cores : {20u, 24u, 32u})
        fallback(16, 320, 4, 2560, false, cores, "Ascend950", 14);
    fallback(1024, 320, 4, 2560, false, 28, "Ascend950", 3);
    fallback(10240, 320, 4, 2560, false, 28, "Ascend950", 8);
    TilingInfo unsupported;
    EXPECT_FALSE(RunSmallModelRouting(16, 320, 4, 2560, false, 28, "Ascend950", ge::DT_FLOAT, true, true, unsupported));
}

TEST(TextGatedResidualTiling, A2FullBSplitKeysExactMetadataAndWorkspace)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t m : {1, 16, 32, 64, 128, 256}) {
            for (bool gate : {false, true}) {
                for (bool gamma : {false, true}) {
                    TilingInfo info;
                    ASSERT_TRUE(
                        RunSmallModelRouting(m, 320, 4, 2560, false, 20, "Ascend910B", dtype, gate, gamma, info));
                    const uint32_t expectedKey = m <= 16 ? 14u : (m <= 64 ? 4u : (m == 128 ? 6u : 7u));
                    EXPECT_EQ(info.tilingKey, expectedKey);
                    EXPECT_EQ(info.blockNum, 20);
                    ASSERT_EQ(info.tilingDataSize, sizeof(HcForwardMulSumTilingData));
                    HcForwardMulSumTilingData t{};
                    std::memcpy(&t, info.tilingData.get(), sizeof(t));
                    EXPECT_EQ(t.mTask, m == 256 ? 128u : 64u);
                    EXPECT_EQ(t.taskNum, m <= 64 ? 1u : 2u);
                    EXPECT_EQ(t.usedAic, 20u);
                    EXPECT_EQ(t.splitkFactor, 10u);
                    EXPECT_EQ(t.smallN, 1u);
                    const int64_t actualC0Slots = 2 * static_cast<int64_t>(t.mTask) * (t.R + (gate ? t.C : 0u)) * t.szT;
                    const int64_t privateEnd =
                        static_cast<int64_t>(t.M) * (2 * t.D + t.R) * t.szT + actualC0Slots + HcSplitkBytes(t);
                    const int64_t expectedUserBytes = HcWorkspaceBytes(t, true);
                    EXPECT_LE(privateEnd, expectedUserBytes);
                    EXPECT_EQ(expectedUserBytes - privateEnd, actualC0Slots);
                    ASSERT_EQ(info.workspaceSizes.size(), 1u);
                    EXPECT_GE(info.workspaceSizes[0], expectedUserBytes);
                }
            }
        }
    }
    for (uint64_t physicalCores : {24u, 32u}) {
        TilingInfo info;
        ASSERT_TRUE(
            RunSmallModelRouting(128, 320, 4, 2560, false, physicalCores, "Ascend910B", ge::DT_BF16, true, true, info));
        EXPECT_EQ(info.tilingKey, 6);
        EXPECT_EQ(info.blockNum, physicalCores);
    }
}

TEST(TextGatedResidualTiling, A2FullBRoutingAndFallback)
{
    auto fallback = [](int64_t m, int64_t r, int64_t c, int64_t h, bool backward, uint64_t cores,
                       int64_t expectedKey = 3) {
        TilingInfo info;
        ASSERT_TRUE(RunSmallModelRouting(m, r, c, h, backward, cores, "Ascend910B", ge::DT_BF16, true, true, info));
        EXPECT_EQ(info.tilingKey, expectedKey);
    };
    fallback(0, 320, 4, 2560, false, 20, 0);
    fallback(2, 320, 4, 2560, false, 20, 14);
    fallback(127, 320, 4, 2560, false, 20, 6);
    for (int64_t m : {129, 255, 257, 384})
        fallback(m, 320, 4, 2560, false, 20, 7);
    for (int64_t m : {385, 512, 8192})
        fallback(m, 320, 4, 2560, false, 20, 3);
    fallback(8193, 320, 4, 2560, false, 20, 9);
    fallback(16, 256, 4, 2560, false, 20);
    fallback(16, 320, 2, 2560, false, 20);
    fallback(16, 320, 4, 2048, false, 20);
    fallback(16, 320, 4, 2560, true, 20, 21);
    fallback(16, 320, 4, 2560, false, 19, 14);
    fallback(16, 320, 4, 2560, false, 15, 3);
    TilingInfo a5OtherCore;
    ASSERT_TRUE(RunSmallModelRouting(16, 320, 4, 2560, false, 20, "Ascend950", ge::DT_BF16, true, true, a5OtherCore));
    EXPECT_EQ(a5OtherCore.tilingKey, 14);
}
