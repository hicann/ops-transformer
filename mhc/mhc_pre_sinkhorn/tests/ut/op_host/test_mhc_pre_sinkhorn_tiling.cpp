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
#include <cstring>
#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/mhc_pre_sinkhorn_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
constexpr char ASCEND950_SOC_VERSION[] = "Ascend950";
constexpr int64_t HC_MULT = 4;
constexpr int64_t HC_MIX = 24;
constexpr int64_t NUM_ITERS = 20;
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

std::string MakeRegbaseSocInfo(uint64_t aicCoreNum, uint64_t aivCoreNum, uint64_t ubSize)
{
    return R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1",)"
           R"("Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,)"
           R"("Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,)"
           R"("L2_SIZE":33554432,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,)"
           R"("UB_SIZE":)" +
           std::to_string(ubSize) + R"(,"CORE_NUM":)" + std::to_string(aicCoreNum) + R"(,"cube_core_cnt":)" +
           std::to_string(aicCoreNum) + R"(,"vector_core_cnt":)" + std::to_string(aivCoreNum) +
           R"(,"socVersion":"Ascend950"}})";
}

struct CoreTilingExpected {
    int64_t rowOfFormerBlock;
    int64_t rowLoopOfFormerBlock;
    int64_t rowLoopOfTailBlock;
    int64_t tailRowFactorOfFormerBlock;
    int64_t tailRowFactorOfTailBlock;
};

struct DTilingExpected {
    int64_t dLoop;
    int64_t dFactor;
    int64_t tailDFactor;
};

struct SplitTilingExpected {
    int64_t kBlockFactor;
    int64_t stage2RowFactor;
    int64_t kL1Size;
    int64_t mL1Size;
    int64_t cubeBlockDimM;
    int64_t cubeBlockDimK;
    int64_t kUbSize;
    int64_t multCoreSplitKSize;
    int64_t secondUsedCoreNum;
    int64_t rowInnerFactor;
    int64_t bufferPool0Size;
    int64_t bufferPool1Size;
    int64_t mUbSize;
};

struct RegbaseTilingExpected {
    CoreTilingExpected core;
    DTilingExpected d;
    SplitTilingExpected split;
};

struct RegbaseTilingBaseline {
    const char* name;
    int64_t sequenceLength;
    int64_t d;
    bool needBackward;
    uint64_t ubSize;
    uint64_t tilingKey;
    size_t workspaceSize;
    RegbaseTilingExpected expected;
    uint64_t coreNum = 64;
    int64_t hcMix = HC_MIX;
};

std::unique_ptr<gert::TilingContextPara> BuildRegbaseTilingContext(const RegbaseTilingBaseline& baseline,
                                                                   void* compileInfo, bool hasPremix = false,
                                                                   int32_t deterministicLevel = 0)
{
    const int64_t b = 1;
    const int64_t s = baseline.sequenceLength;
    const int64_t phiDim1 = HC_MULT * baseline.d;
    std::unique_ptr<gert::TilingContextPara> context(new gert::TilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{b, s, HC_MULT, baseline.d}, {b, s, HC_MULT, baseline.d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{baseline.hcMix, phiDim1}, {baseline.hcMix, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{baseline.hcMix}, {baseline.hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{b, s, baseline.d}, {b, s, baseline.d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{b, s, HC_MULT}, {b, s, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, s, HC_MULT, HC_MULT}, {b, s, HC_MULT, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, s, HC_MULT}, {b, s, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, s, baseline.hcMix}, {b, s, baseline.hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, s, 1}, {b, s, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{NUM_ITERS * 2, b, s, HC_MULT}, {NUM_ITERS * 2, b, s, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{NUM_ITERS * 2, b, s, HC_MULT, HC_MULT}, {NUM_ITERS * 2, b, s, HC_MULT, HC_MULT}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(HC_MULT)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(NUM_ITERS)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(baseline.needBackward)},
        },
        compileInfo, ASCEND950_SOC_VERSION, baseline.coreNum, baseline.ubSize));
    if (hasPremix) {
        context->inputTensorDesc_.push_back({{{b, s, HC_MULT}, {b, s, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND});
        context->inputInstanceNum_ = {1, 1, 1, 1, 1};
        context->outputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1};
    }
    context->deterministicInfo_ = deterministicLevel;
    return context;
}

template <typename T>
T ReadTilingField(const uint8_t* buffer, size_t& offset)
{
    T value = 0;
    std::memcpy(&value, buffer + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

void DeserializeRegbaseTilingData(const TilingInfo& tilingInfo, optiling::MhcPreSinkhornRegbaseTilingData& tilingData)
{
    size_t offset = 0;
#define READ_TILING_FIELD(type, field) \
    tilingData.set_##field(ReadTilingField<type>(tilingInfo.tilingData.get(), offset))
    READ_TILING_FIELD(int64_t, bs);
    READ_TILING_FIELD(int64_t, hcMix);
    READ_TILING_FIELD(int64_t, hcMult);
    READ_TILING_FIELD(int64_t, d);
    READ_TILING_FIELD(int64_t, hcMultAlign);
    READ_TILING_FIELD(int64_t, rowOfFormerBlock);
    READ_TILING_FIELD(int64_t, rowLoopOfFormerBlock);
    READ_TILING_FIELD(int64_t, rowLoopOfTailBlock);
    READ_TILING_FIELD(int64_t, tailRowFactorOfFormerBlock);
    READ_TILING_FIELD(int64_t, tailRowFactorOfTailBlock);
    READ_TILING_FIELD(int64_t, dLoop);
    READ_TILING_FIELD(int64_t, dFactor);
    READ_TILING_FIELD(int64_t, tailDFactor);
    READ_TILING_FIELD(int64_t, iterTimes);
    READ_TILING_FIELD(float, hcEps);
    READ_TILING_FIELD(float, normEps);
    READ_TILING_FIELD(int64_t, kBlockFactor);
    READ_TILING_FIELD(int64_t, stage2RowFactor);
    READ_TILING_FIELD(int64_t, k);
    READ_TILING_FIELD(int64_t, kL1Size);
    READ_TILING_FIELD(int64_t, mL1Size);
    READ_TILING_FIELD(int64_t, cubeBlockDimM);
    READ_TILING_FIELD(int64_t, cubeBlockDimK);
    READ_TILING_FIELD(int64_t, kUbSize);
    READ_TILING_FIELD(int64_t, multCoreSplitKSize);
    READ_TILING_FIELD(int64_t, secondUsedCoreNum);
    READ_TILING_FIELD(int64_t, rowInnerFactor);
    READ_TILING_FIELD(int64_t, bufferPool0Size);
    READ_TILING_FIELD(int64_t, bufferPool1Size);
    READ_TILING_FIELD(int64_t, mUbSize);
    READ_TILING_FIELD(int64_t, sinkhornCoreNum);
    READ_TILING_FIELD(int64_t, sinkhornRowFactor);
    READ_TILING_FIELD(int64_t, hasPremix);
    READ_TILING_FIELD(int64_t, premixCastSegments);
    READ_TILING_FIELD(int64_t, premixStage1);
#undef READ_TILING_FIELD
    EXPECT_EQ(offset, tilingInfo.tilingDataSize) << "Regbase tiling data schema changed; update the deserializer";
}

bool ExecutePremixTiling(const RegbaseTilingBaseline& baseline, uint64_t ratio,
                         optiling::MhcPreSinkhornRegbaseTilingData& actual)
{
    optiling::MhcPreSinkhornCompileInfo compileInfo = {baseline.coreNum * ratio, baseline.coreNum, baseline.ubSize,
                                                       SYS_WORKSPACE_SIZE, ratio};
    auto context = BuildRegbaseTilingContext(baseline, &compileInfo, true);
    context->socInfoString_ = MakeRegbaseSocInfo(baseline.coreNum, baseline.coreNum * ratio, baseline.ubSize);
    TilingInfo info;
    if (!ExecuteTiling(*context, info)) {
        return false;
    }
    DeserializeRegbaseTilingData(info, actual);
    return true;
}

void ExpectRegbaseTilingData(const RegbaseTilingBaseline& baseline)
{
    TilingInfo tilingInfo;
    // pin cvRatio=2 explicitly for the Ascend950 baselines.
    optiling::MhcPreSinkhornCompileInfo compileInfo = {baseline.coreNum, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, 2};
    auto tilingContext = BuildRegbaseTilingContext(baseline, &compileInfo);
    ASSERT_TRUE(ExecuteTiling(*tilingContext, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, baseline.tilingKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], static_cast<int64_t>(baseline.workspaceSize));
    optiling::MhcPreSinkhornRegbaseTilingData actual;
    ASSERT_EQ(tilingInfo.tilingDataSize, actual.GetDataSize());
    DeserializeRegbaseTilingData(tilingInfo, actual);

#define EXPECT_TILING_FIELD(field, expected) EXPECT_EQ(actual.get_##field(), (expected)) << baseline.name << ": " #field
    EXPECT_TILING_FIELD(hasPremix, 0);
    EXPECT_TILING_FIELD(premixCastSegments, 0);
    EXPECT_TILING_FIELD(premixStage1, 0);
    EXPECT_TILING_FIELD(bs, baseline.sequenceLength);
    EXPECT_TILING_FIELD(hcMix, baseline.hcMix);
    EXPECT_TILING_FIELD(hcMult, HC_MULT);
    EXPECT_TILING_FIELD(d, baseline.d);
    EXPECT_TILING_FIELD(hcMultAlign, 8);
    EXPECT_TILING_FIELD(rowOfFormerBlock, baseline.expected.core.rowOfFormerBlock);
    EXPECT_TILING_FIELD(rowLoopOfFormerBlock, baseline.expected.core.rowLoopOfFormerBlock);
    EXPECT_TILING_FIELD(rowLoopOfTailBlock, baseline.expected.core.rowLoopOfTailBlock);
    EXPECT_TILING_FIELD(tailRowFactorOfFormerBlock, baseline.expected.core.tailRowFactorOfFormerBlock);
    EXPECT_TILING_FIELD(tailRowFactorOfTailBlock, baseline.expected.core.tailRowFactorOfTailBlock);
    EXPECT_TILING_FIELD(dLoop, baseline.expected.d.dLoop);
    EXPECT_TILING_FIELD(dFactor, baseline.expected.d.dFactor);
    EXPECT_TILING_FIELD(tailDFactor, baseline.expected.d.tailDFactor);
    EXPECT_TILING_FIELD(iterTimes, NUM_ITERS);
    EXPECT_FLOAT_EQ(actual.get_hcEps(), 1e-6f) << baseline.name << ": hcEps";
    EXPECT_FLOAT_EQ(actual.get_normEps(), 1e-6f) << baseline.name << ": normEps";
    EXPECT_TILING_FIELD(kBlockFactor, baseline.expected.split.kBlockFactor);
    EXPECT_TILING_FIELD(stage2RowFactor, baseline.expected.split.stage2RowFactor);
    EXPECT_TILING_FIELD(k, HC_MULT * baseline.d);
    EXPECT_TILING_FIELD(kL1Size, baseline.expected.split.kL1Size);
    EXPECT_TILING_FIELD(mL1Size, baseline.expected.split.mL1Size);
    EXPECT_TILING_FIELD(cubeBlockDimM, baseline.expected.split.cubeBlockDimM);
    EXPECT_TILING_FIELD(cubeBlockDimK, baseline.expected.split.cubeBlockDimK);
    EXPECT_TILING_FIELD(kUbSize, baseline.expected.split.kUbSize);
    EXPECT_TILING_FIELD(multCoreSplitKSize, baseline.expected.split.multCoreSplitKSize);
    EXPECT_TILING_FIELD(secondUsedCoreNum, baseline.expected.split.secondUsedCoreNum);
    EXPECT_TILING_FIELD(rowInnerFactor, baseline.expected.split.rowInnerFactor);
    EXPECT_TILING_FIELD(bufferPool0Size, baseline.expected.split.bufferPool0Size);
    EXPECT_TILING_FIELD(bufferPool1Size, baseline.expected.split.bufferPool1Size);
    EXPECT_TILING_FIELD(mUbSize, baseline.expected.split.mUbSize);
#undef EXPECT_TILING_FIELD
}

void ExpectRegbaseTilingFailure(const RegbaseTilingBaseline& baseline)
{
    TilingInfo tilingInfo;
    // pin cvRatio=2 explicitly for the Ascend950 baselines.
    optiling::MhcPreSinkhornCompileInfo compileInfo = {baseline.coreNum, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, 2};
    auto tilingContext = BuildRegbaseTilingContext(baseline, &compileInfo);
    EXPECT_FALSE(ExecuteTiling(*tilingContext, tilingInfo));
}
} // namespace

class MhcPreSinkhornTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSinkhornTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSinkhornTiling TearDown" << std::endl;
    }
};

TEST_F(MhcPreSinkhornTiling, DAV_3510_regbase_tilingdata_matches_pre_refactor)
{
    // The first four cases cover the original functions: K/M split x no-grad/grad-out; the last covers D splitting.
    const std::array<RegbaseTilingBaseline, 5> baselines = {{
        {"k_split_no_grad",
         256,
         4096,
         false,
         262144,
         1000,
         18415616,
         {{5, 3, 1, 1, 1}, {1, 4096, 4096}, {64, 2, 128, 256, 1, 64, 64, 256, 52, 0, 0, 0, 128}}},
        {"m_split_no_grad",
         16384,
         4096,
         false,
         262144,
         1001,
         SYS_WORKSPACE_SIZE,
         {{256, 86, 86, 1, 1}, {1, 4096, 4096}, {1, 0, 128, 256, 64, 1, 64, 16384, 0, 3, 262144, 249152, 128}}},
        {"k_split_grad_out",
         256,
         7168,
         true,
         262144,
         2000,
         18210816,
         {{4, 4, 4, 1, 1}, {1, 7168, 7168}, {56, 1, 128, 256, 1, 56, 64, 512, 64, 0, 0, 0, 128}}},
        {"m_split_grad_out",
         16384,
         7168,
         true,
         262144,
         2001,
         SYS_WORKSPACE_SIZE,
         {{256, 256, 256, 1, 1}, {1, 7168, 7168}, {1, 0, 128, 256, 64, 1, 64, 28672, 0, 1, 262144, 249152, 128}}},
        {"k_split_no_grad_d_split",
         256,
         4096,
         false,
         49152,
         1000,
         18415616,
         {{5, 5, 1, 1, 1}, {4, 1344, 64}, {64, 1, 128, 256, 1, 64, 64, 256, 52, 0, 0, 0, 128}}},
    }};

    for (const auto& baseline : baselines) {
        SCOPED_TRACE(baseline.name);
        ExpectRegbaseTilingData(baseline);
    }
}

TEST_F(MhcPreSinkhornTiling, DAV_3510_regbase_core_tiling_boundary_branches)
{
    // These cases cover Ascend950-only common tiling boundaries: partial core rows, the dFactor path without
    // DownAlign, and M-split grad-out UB accounting. The preceding regression case covers dFactor DownAlign.
    const std::array<RegbaseTilingBaseline, 3> baselines = {{
        {"k_split_partial_core_rows",
         65,
         4096,
         false,
         262144,
         1000,
         17193216,
         {{2, 1, 1, 2, 1}, {1, 4096, 4096}, {64, 2, 384, 80, 1, 64, 192, 256, 33, 0, 0, 0, 40}}},
        {"k_split_d_factor_at_ub_limit",
         256,
         4096,
         false,
         17312,
         1000,
         18415616,
         {{5, 5, 1, 1, 1}, {128, 32, 32}, {64, 1, 128, 256, 1, 64, 64, 256, 52, 0, 0, 0, 128}}},
        {"m_split_grad_out_d_factor_at_ub_limit",
         16384,
         4096,
         true,
         51232,
         2001,
         SYS_WORKSPACE_SIZE,
         {{256, 256, 256, 1, 1}, {256, 16, 16}, {1, 0, 128, 256, 64, 1, 64, 16384, 0, 1, 51232, 38240, 128}}},
    }};

    for (const auto& baseline : baselines) {
        SCOPED_TRACE(baseline.name);
        ExpectRegbaseTilingData(baseline);
    }

    const RegbaseTilingBaseline insufficientUb = {
        "k_split_insufficient_minimum_tile", 256, 4096, false, 16927, 0, 0, {}};
    ExpectRegbaseTilingFailure(insufficientUb);

    RegbaseTilingBaseline insufficientMBufferPool = {
        "m_split_negative_buffer_pool", 16384, 4096, false, 262144, 0, 0, {}};
    insufficientMBufferPool.hcMix = 1024;
    ExpectRegbaseTilingFailure(insufficientMBufferPool);
}

TEST_F(MhcPreSinkhornTiling, ascend950_premix_cast_segments_and_ub_fallback)
{
    for (bool needBackward : {false, true}) {
        // The four cast slices require more UB when CV ratio 1 gives each AIV the full M tile.
        for (uint64_t cvRatio : {1U, 2U}) {
            for (uint64_t ubSize : {131072U, 655360U}) {
                SCOPED_TRACE(::testing::Message() << needBackward << ":" << cvRatio << ":" << ubSize);
                RegbaseTilingBaseline baseline = {"premix_m_split", 16385, 4096, needBackward, ubSize, 0, 0, {}};
                optiling::MhcPreSinkhornCompileInfo compileInfo = {baseline.coreNum, 0, ubSize, SYS_WORKSPACE_SIZE,
                                                                   cvRatio};
                auto context = BuildRegbaseTilingContext(baseline, &compileInfo, true);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(*context, info));
                optiling::MhcPreSinkhornRegbaseTilingData actual;
                DeserializeRegbaseTilingData(info, actual);
                EXPECT_EQ(actual.get_cubeBlockDimK(), 1);
                EXPECT_EQ(actual.get_hasPremix(), 1);
                EXPECT_EQ(actual.get_premixCastSegments(), ubSize == 655360U ? 1 : 0);
                if (actual.get_premixCastSegments() != 0) {
                    EXPECT_EQ(actual.get_mUbSize(), (actual.get_mL1Size() + cvRatio - 1) / cvRatio);
                    EXPECT_EQ(baseline.d % actual.get_kUbSize(), 0);
                }
            }
        }
    }
}

TEST_F(MhcPreSinkhornTiling, ascend950_premix_k_split_compares_both_windows)
{
    // Both windows fit, but the half window handles more M rows per chunk.
    const RegbaseTilingBaseline baseline = {"premix_k_split", 256, 4096, true, 98304, 0, 0, {}};
    optiling::MhcPreSinkhornCompileInfo compileInfo = {64, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, 2};
    auto context = BuildRegbaseTilingContext(baseline, &compileInfo, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(*context, info));
    optiling::MhcPreSinkhornRegbaseTilingData actual;
    DeserializeRegbaseTilingData(info, actual);
    EXPECT_GT(actual.get_cubeBlockDimK(), 1);
    EXPECT_EQ(actual.get_kUbSize(), 32);
    EXPECT_EQ(actual.get_premixCastSegments(), 0);
}

TEST_F(MhcPreSinkhornTiling, ascend950_premix_k_split_selects_parallel_stage2)
{
    for (uint64_t cores : {28U, 32U}) {
        for (uint64_t ratio : {1U, 2U}) {
            for (int64_t rows : {1, 32, 128, 192}) {
                for (bool needBackward : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << cores << ":" << ratio << ":" << rows << ":" << needBackward);
                    RegbaseTilingBaseline baseline = {
                        "premix_k_accumulator", rows, 4096, needBackward, 253952, 0, 0, {}};
                    baseline.coreNum = cores;
                    optiling::MhcPreSinkhornRegbaseTilingData actual;
                    ASSERT_TRUE(ExecutePremixTiling(baseline, ratio, actual));
                    EXPECT_GT(actual.get_cubeBlockDimK(), 1);
                    const int64_t expectedStage1 = !needBackward && ratio == 2 && rows >= 128 ? 0 : 1;
                    EXPECT_EQ(actual.get_premixStage1(), expectedStage1);
                    EXPECT_EQ(actual.get_premixCastSegments(), 0);
                }
            }
        }
    }
}

TEST_F(MhcPreSinkhornTiling, ascend950_premix_stage_selection_respects_parallelism_and_ub)
{
    struct PremixCase {
        int64_t rows;
        int64_t d;
        uint64_t cores;
        uint64_t ratio;
        uint64_t ubSize;
        bool inferenceStage1;
        bool backwardSupported;
    };
    const std::array<PremixCase, 8> cases = {{{1, 2048, 28, 2, 253952, true, true},
                                              {32, 4096, 28, 2, 253952, true, true},
                                              {128, 4096, 32, 2, 253952, false, true},
                                              {192, 7168, 32, 2, 253952, false, true},
                                              {511, 1024, 32, 2, 253952, true, true},
                                              {32, 99968, 24, 1, 98304, false, false},
                                              {192, 2048, 28, 2, 65536, true, true},
                                              {192, 2048, 28, 1, 253952, true, true}}};
    for (const auto& test : cases) {
        for (bool needBackward : {false, true}) {
            SCOPED_TRACE(::testing::Message() << test.rows << ":" << test.d << ":" << needBackward);
            RegbaseTilingBaseline baseline = {"premix_ub_fallback", test.rows, test.d, needBackward,
                                              test.ubSize,          0,         0,      {}};
            baseline.coreNum = test.cores;
            optiling::MhcPreSinkhornRegbaseTilingData actual;
            if (!test.backwardSupported && needBackward) {
                EXPECT_FALSE(ExecutePremixTiling(baseline, test.ratio, actual));
                continue;
            }
            ASSERT_TRUE(ExecutePremixTiling(baseline, test.ratio, actual));
            EXPECT_GT(actual.get_cubeBlockDimK(), 1);
            EXPECT_EQ(actual.get_hasPremix(), 1);
            EXPECT_EQ(actual.get_premixStage1(), needBackward || test.inferenceStage1 ? 1 : 0);
        }
    }
}

TEST_F(MhcPreSinkhornTiling, ascend950_m_partition_matches_with_and_without_premix)
{
    struct SplitCase {
        uint64_t cores;
        uint64_t cvRatio;
        int64_t rows;
        int64_t d;
        int32_t deterministicLevel;
    };
    // Cover M+K split, M-only split, multiple M rounds, and batch consistency (level 3).
    const std::array<SplitCase, 6> cases = {{{28, 2, 3072, 1792, 0},
                                             {32, 1, 4096, 4096, 0},
                                             {28, 2, 4096, 4096, 0},
                                             {32, 2, 8193, 2048, 0},
                                             {28, 2, 65536, 7168, 0},
                                             {32, 2, 65, 4096, 3}}};
    for (const auto& test : cases) {
        for (bool needBackward : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << test.cores << ":" << test.cvRatio << ":" << test.rows << ":" << needBackward);
            RegbaseTilingBaseline baseline = {
                "m_partition_parity", test.rows, test.d, needBackward, 253952, 0, 0, {}, test.cores};
            const uint64_t vectorCores = test.cores * test.cvRatio;
            optiling::MhcPreSinkhornCompileInfo compileInfo = {vectorCores, test.cores, baseline.ubSize,
                                                               SYS_WORKSPACE_SIZE, test.cvRatio};
            auto withoutPremix = BuildRegbaseTilingContext(baseline, &compileInfo, false, test.deterministicLevel);
            auto withPremix = BuildRegbaseTilingContext(baseline, &compileInfo, true, test.deterministicLevel);
            withoutPremix->socInfoString_ = MakeRegbaseSocInfo(test.cores, vectorCores, baseline.ubSize);
            withPremix->socInfoString_ = withoutPremix->socInfoString_;

            TilingInfo baselineInfo;
            TilingInfo premixInfo;
            ASSERT_TRUE(ExecuteTiling(*withoutPremix, baselineInfo));
            ASSERT_TRUE(ExecuteTiling(*withPremix, premixInfo));
            EXPECT_EQ(premixInfo.tilingKey, baselineInfo.tilingKey);
            EXPECT_EQ(premixInfo.blockNum, baselineInfo.blockNum);
            EXPECT_EQ(premixInfo.workspaceSizes, baselineInfo.workspaceSizes);

            optiling::MhcPreSinkhornRegbaseTilingData expected;
            optiling::MhcPreSinkhornRegbaseTilingData actual;
            DeserializeRegbaseTilingData(baselineInfo, expected);
            DeserializeRegbaseTilingData(premixInfo, actual);
            EXPECT_EQ(actual.get_hasPremix(), 1);
            EXPECT_EQ(actual.get_cubeBlockDimM(), expected.get_cubeBlockDimM());
            EXPECT_EQ(actual.get_cubeBlockDimK(), expected.get_cubeBlockDimK());
            EXPECT_EQ(actual.get_mL1Size(), expected.get_mL1Size());
            EXPECT_EQ(actual.get_kL1Size(), expected.get_kL1Size());
            EXPECT_EQ(actual.get_multCoreSplitKSize(), expected.get_multCoreSplitKSize());
        }
    }
}

TEST_F(MhcPreSinkhornTiling, ascend950_premix_m_split_uses_partial_ub_rows)
{
    for (int64_t cores : {28, 32}) {
        for (bool needBackward : {false, true}) {
            RegbaseTilingBaseline baseline = {"premix_partial_m_ub", cores * 256, 1792, needBackward, 131072, 0, 0, {}};
            baseline.coreNum = cores;
            optiling::MhcPreSinkhornRegbaseTilingData actual;
            ASSERT_TRUE(ExecutePremixTiling(baseline, 2, actual));
            EXPECT_EQ(actual.get_cubeBlockDimM(), cores);
            EXPECT_EQ(actual.get_cubeBlockDimK(), 1);
            EXPECT_EQ(actual.get_mL1Size(), 256);
            EXPECT_EQ(actual.get_premixCastSegments(), 0);
            EXPECT_GT(actual.get_mUbSize(), 0);
            EXPECT_LT(actual.get_mUbSize(), 128);
        }
    }
}

// A2 tiling期望数据说明: premix合并新增了 prePost*/sinkhorn* 17个MPMD计算字段与 hasPremix
// 开关字段, enable_hf32合入新增 isHf32 开关字段; 其中17个MPMD字段由新tiling算法在宿主环境
// 算出, 此处以框架支持的 '*' 掩码跳过; hasPremix 固定为0; isHf32 因 enable_hf32 缺省为true(与premix合入前一致),
// 固定为1; 其余字段值与合并前基线一致, 新增大K分段字段后共123个token(98标量+25 mm1TilingData)。
TEST_F(MhcPreSinkhornTiling, test_tiling_B1_S1024_N4_C512)
{
    int64_t B = 1;
    int64_t S = 1024;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N}, {numIters * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N, N}, {numIters * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 1;
    string expectTilingData =
        "1024 24 4 512 8 16 16 16 16 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 "
        "128 256 4 8 0 1024 256 256 0 0 0 0 0 0 0 0 64 0 64 21 0 0 0 0 0 16 16 16 16 1 1 1 0 0 0 0 "
        "0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 16 16 1 32 8 256 256 1 2048 33554432 "
        "274877906945 8796093022232 274877908992 8796093022232 137438953536 8589934720 4294967298 "
        "1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {251658240};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_3d_S1024_N4_C512)
{
    int64_t T = 1024;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, T, N}, {numIters * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, T, N, N}, {numIters * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 1;
    string expectTilingData =
        "1024 24 4 512 8 16 16 16 16 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 "
        "128 256 4 8 0 1024 256 256 0 0 0 0 0 0 0 0 64 0 64 21 0 0 0 0 0 16 16 16 16 1 1 1 0 0 0 0 "
        "0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 16 16 1 32 8 256 256 1 2048 33554432 "
        "274877906945 8796093022232 274877908992 8796093022232 137438953536 8589934720 4294967298 "
        "1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {251658240};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_B2_S128_N4_C256)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N}, {numIters * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N, N}, {numIters * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 1;
    string expectTilingData =
        "256 24 4 256 8 4 4 4 4 0 1 1 1 256 256 20 3856831416675743677 0 0 0 0 1 1024 0 0 0 0 128 "
        "256 1 4 0 1024 256 256 0 0 0 0 0 0 0 0 64 0 64 21 0 0 0 0 0 4 4 4 4 1 1 1 0 0 0 0 0 0 0 0 "
        "1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 4 4 1 32 4 256 256 1 1024 16777216 274877906945 "
        "4398046511128 274877907968 4398046511128 137438953536 8589934720 4294967304 1 0 "
        "844424930131968 8192 4294967297 4294967297 34359738369 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {234881024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_no_backward_B1_S256_N4_C512)
{
    int64_t B = 1;
    int64_t S = 256;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingData =
        "256 24 4 512 8 22 14 2 2 0 11 3 1 512 512 20 3856831416675743677 0 0 0 0 11 2048 0 0 0 0 "
        "128 256 1 8 0 1024 256 256 0 0 0 0 0 0 0 0 12 0 64 21 0 0 0 12 52 22 14 6 4 4 2 2 52 5 1 "
        "1 1 5 5 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 4 4 1 32 8 256 256 1 2048 33554432 "
        "274877906945 8796093022232 274877908992 8796093022232 137438953536 8589934720 4294967298 "
        "1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {251683840};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_premix_B1_S256_N4_C512)
{
    // membase(A2/A3) 传入可选 premix 输入: hasPremix=1, tiling 成功, 走 KEY=1(m_k original)
    int64_t B = 1;
    int64_t S = 256;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N}, {numIters * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N, N}, {numIters * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 1;
    // 期望串为占位, 首次远程实跑后按实际输出回填基线
    string expectTilingData =
        "256 24 4 512 8 4 4 4 4 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 128 "
        "256 1 8 0 1024 256 256 0 0 0 0 0 0 0 0 64 0 64 21 0 0 0 0 0 4 4 4 4 1 1 1 0 0 0 0 0 0 0 0 "
        "1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 4 4 1 32 8 256 256 1 2048 33554432 274877906945 "
        "8796093022232 274877908992 8796093022232 137438953536 8589934720 4294967298 1 0 "
        "422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {251658240};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, pre_mix_supports_both_splits_and_cv_ratios)
{
    for (uint64_t ratio : {1U, 2U}) {
        for (bool needBackward : {false, true}) {
            for (int64_t rows : {256, 16384}) {
                SCOPED_TRACE(testing::Message()
                             << "ratio=" << ratio << ", backward=" << needBackward << ", rows=" << rows);
                RegbaseTilingBaseline baseline = {};
                baseline.sequenceLength = rows;
                baseline.d = 4096;
                baseline.needBackward = needBackward;
                baseline.ubSize = 262144;
                optiling::MhcPreSinkhornCompileInfo compileInfo = {64, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, ratio};
                auto context = BuildRegbaseTilingContext(baseline, &compileInfo);
                context->inputTensorDesc_.push_back(
                    {{{1, rows, HC_MULT}, {1, rows, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND});
                context->inputInstanceNum_ = {1, 1, 1, 1, 1};
                context->outputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1};
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(*context, info));
                EXPECT_EQ(info.tilingKey,
                          (ratio == 1 ? 3000U : 1000U) + (needBackward ? 1000U : 0U) + (rows == 16384 ? 1U : 0U));
                optiling::MhcPreSinkhornRegbaseTilingData data;
                DeserializeRegbaseTilingData(info, data);
                EXPECT_EQ(data.get_hasPremix(), 1);
                EXPECT_GT(data.get_dFactor(), 0);
                EXPECT_GT(rows == 16384 ? data.get_rowInnerFactor() : data.get_stage2RowFactor(), 0);
            }
        }
    }
}

TEST_F(MhcPreSinkhornTiling, pre_mix_rejects_mismatched_leading_dimension_and_dtype)
{
    RegbaseTilingBaseline baseline = {};
    baseline.sequenceLength = 256;
    baseline.d = 4096;
    baseline.ubSize = 262144;
    optiling::MhcPreSinkhornCompileInfo compileInfo = {64, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, 2};
    for (bool badDtype : {false, true}) {
        auto context = BuildRegbaseTilingContext(baseline, &compileInfo);
        const int64_t rows = badDtype ? 256 : 255;
        context->inputTensorDesc_.push_back(
            {{{1, rows, HC_MULT}, {1, rows, HC_MULT}}, badDtype ? ge::DT_FLOAT16 : ge::DT_FLOAT, ge::FORMAT_ND});
        context->inputInstanceNum_ = {1, 1, 1, 1, 1};
        context->outputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1};
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(*context, info));
    }
}

TEST_F(MhcPreSinkhornTiling, pre_mix_supports_tnd)
{
    RegbaseTilingBaseline baseline = {};
    baseline.sequenceLength = 256;
    baseline.d = 4096;
    baseline.ubSize = 262144;
    optiling::MhcPreSinkhornCompileInfo compileInfo = {64, 0, baseline.ubSize, SYS_WORKSPACE_SIZE, 2};
    auto context = BuildRegbaseTilingContext(baseline, &compileInfo);
    context->inputTensorDesc_[0] = {{{256, HC_MULT, 4096}, {256, HC_MULT, 4096}}, ge::DT_BF16, ge::FORMAT_ND};
    context->inputTensorDesc_.push_back({{{256, HC_MULT}, {256, HC_MULT}}, ge::DT_FLOAT, ge::FORMAT_ND});
    context->inputInstanceNum_ = {1, 1, 1, 1, 1};
    context->outputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1};
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(*context, info));
    optiling::MhcPreSinkhornRegbaseTilingData data;
    DeserializeRegbaseTilingData(info, data);
    EXPECT_EQ(data.get_hasPremix(), 1);
    EXPECT_EQ(data.get_bs(), 256);
}
