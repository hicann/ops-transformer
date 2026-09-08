/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstring>
#include <gtest/gtest.h>
#include "../../../../op_host/op_tiling/arch22/grouped_mat_mul_allto_allv_mte_tiling.h"
#include "../../../../op_kernel/arch22/grouped_mat_mul_allto_allv_mte_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace GroupedMatMulAlltoAllvMteUT {

constexpr uint64_t DEFAULT_CCL_BUFFER_SIZE = 6000ULL * 1024ULL * 1024ULL;
constexpr uint64_t SAMPLE_OUTPUT_SIZE = 4096ULL * 4096ULL * sizeof(uint16_t);
constexpr uint64_t SAMPLE_COUNT_MATRIX_SIZE = 8ULL * 4ULL * sizeof(int32_t);
constexpr uint64_t SAMPLE_REQUIRED_CCL_BUFFER_SIZE = 200ULL * 1024ULL * 1024ULL;

struct TestParam {
    string testName{};
    std::vector<std::pair<string, string>> tilingParamsStrPair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tilingParamsVecPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingDTypesPair{};
    ge::graphStatus status;
    uint64_t cclBufferSize{DEFAULT_CCL_BUFFER_SIZE};
    uint64_t communicatorRankSize{8UL};
    std::string socVersion{"Ascend910B"};
    uint32_t expectedChunkRows{0U};
    ge::DataType dtype{ge::DT_BF16};
    std::string commMode{"aiv"};
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
    uint64_t epWorldSize{8};
    uint64_t e{4};
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t gmmWeightDim1{7168};
    uint64_t yDim1{4096};
    uint64_t mmWeightDim0{7168};
    bool transGmmWeight{false};
    bool transMmWeight{false};
    std::string group{"group"};
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};

static std::vector<int64_t> MakeCounts(size_t countNum, int64_t totalCount)
{
    std::vector<int64_t> counts(countNum, totalCount / static_cast<int64_t>(countNum));
    for (size_t i = 0; i < static_cast<size_t>(totalCount % static_cast<int64_t>(countNum)); ++i) {
        ++counts[i];
    }
    return counts;
}

std::unordered_map<string, std::function<void(TilingParams &tilingParams, const string &valueStr)>>
    g_tilingParamsStrHandlers = {
        {"BSK", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.BSK = std::stoi(valueStr); }},
        {"BS", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.BS = std::stoi(valueStr); }},
        {"K", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.K = std::stoi(valueStr); }},
        {"H1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.H1 = std::stoi(valueStr); }},
        {"H2", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.H2 = std::stoi(valueStr); }},
        {"A", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.A = std::stoi(valueStr); }},
        {"N1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.N1 = std::stoi(valueStr); }},
        {"N2", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.N2 = std::stoi(valueStr); }},
        {"epWorldSize",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.epWorldSize = std::stoi(valueStr); }},
        {"e", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.e = std::stoi(valueStr); }},
        {"gmmWeightDim1",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.gmmWeightDim1 = std::stoi(valueStr); }},
        {"yDim1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.yDim1 = std::stoi(valueStr); }},
        {"mmWeightDim0",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.mmWeightDim0 = std::stoi(valueStr); }},
        {"transGmmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transGmmWeight = valueStr == "true"; }},
        {"transMmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transMmWeight = valueStr == "true"; }}};

std::unordered_map<string, std::function<void(TilingParams &tilingParams, const std::vector<int64_t> valueVec)>>
    g_tilingParamsVecHandlers = {
        {"sendCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.sendCounts = valueVec; }},
        {"recvCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.recvCounts = valueVec; }}};

class GroupedMatMulAlltoAllvMteArch22TilingTest : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvMteArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvMteArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, FixedSyncLayout)
{
    using namespace MC2KernelTemplate::Gmma2avMteTiling;
    EXPECT_EQ(EPOCH_BASE, 0UL);
    EXPECT_EQ(COUNT_READY_BASE, SYNC_SLOT_BYTES);
    EXPECT_EQ(EXPERT_READY_BASE, COUNT_READY_BASE + static_cast<uint64_t>(MAX_RANK_SIZE) * SYNC_SLOT_BYTES);
    EXPECT_EQ(COMPLETION_BASE, EXPERT_READY_BASE + static_cast<uint64_t>(MAX_COUNT_NUM) * SYNC_SLOT_BYTES);
    EXPECT_EQ(ACK_BASE, COMPLETION_BASE + static_cast<uint64_t>(MAX_RANK_SIZE) * SYNC_SLOT_BYTES);
    EXPECT_EQ(FIXED_SYNC_BYTES, (1UL + 3UL * MAX_RANK_SIZE + MAX_COUNT_NUM) * SYNC_SLOT_BYTES);
    EXPECT_LE(FIXED_SYNC_BYTES, SYNC_REGION_FROM_TAIL);
}

TEST_P(GroupedMatMulAlltoAllvMteArch22TilingTest, ShapeSize)
{
    auto testParam = GetParam();
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = testParam.socVersion;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    auto tilingParams = TilingParams{};
    for (auto &kv : testParam.tilingParamsStrPair) {
        if (g_tilingParamsStrHandlers.count(kv.first) != 0) {
            g_tilingParamsStrHandlers[kv.first](tilingParams, kv.second);
        }
    }
    for (auto &kv : testParam.tilingParamsVecPair) {
        if (g_tilingParamsVecHandlers.count(kv.first) != 0) {
            g_tilingParamsVecHandlers[kv.first](tilingParams, kv.second);
        }
    }

    // Slots 0..5 are inputs; slots 6..7 are GMM/shared-MM outputs.
    std::vector<ge::DataType> dtypes(8, testParam.dtype);
    for (const auto &item : testParam.tilingDTypesPair) {
        dtypes.at(item.first) = item.second;
    }

    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, dtypes[0], ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1},
              {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1}},
             dtypes[1],
             ge::FORMAT_ND},
            {{}, dtypes[2], ge::FORMAT_ND},
            {{}, dtypes[3], ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}}, dtypes[4], ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim0, tilingParams.N2}, {tilingParams.mmWeightDim0, tilingParams.N2}},
             dtypes[5],
             ge::FORMAT_ND},
        },
        {
            {{{tilingParams.BSK, tilingParams.yDim1}, {tilingParams.BSK, tilingParams.yDim1}},
             dtypes[6],
             ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.N2}, {tilingParams.BS, tilingParams.N2}}, dtypes[7], ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>(testParam.commMode)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", testParam.communicatorRankSize},
                                                   {"cclBufferSize", testParam.cclBufferSize}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", tilingParams.epWorldSize},
                                                   {"cclBufferSize", testParam.cclBufferSize}};
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo{};
        const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        ASSERT_TRUE(tilingSucceeded);
        EXPECT_EQ(tilingInfo.tilingKey, 17UL);
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData));
        MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData actualTiling{};
        std::memcpy(&actualTiling, tilingInfo.tilingData.get(), sizeof(actualTiling));
        EXPECT_EQ(actualTiling.reserved, 0U);
        EXPECT_EQ(actualTiling.expertChunkRows, testParam.expectedChunkRows);
    }
}

static TestParam g_testParams[] = {
    {"Test_a3_topk_16_experts_1024",
     {{"BS", "256"}, {"epWorldSize", "128"}, {"e", "8"}},
     {{"sendCounts", MakeCounts(1024, 4096)}, {"recvCounts", MakeCounts(1024, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     128UL,
     "Ascend910_93",
     1536U},
    {"Test_a3_topk_16_local_experts_512",
     {{"BS", "256"}, {"epWorldSize", "2"}, {"e", "512"}},
     {{"sendCounts", MakeCounts(1024, 4096)}, {"recvCounts", MakeCounts(1024, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     2UL,
     "Ascend910_93",
     1536U},
    {"Test_a3_topk_9",
     {{"BSK", "4608"}, {"BS", "512"}},
     {{"recvCounts", MakeCounts(32, 4608)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     1536U},
    {"Test_a3_topk_17_rejected",
     {{"BSK", "4352"}, {"BS", "256"}},
     {{"recvCounts", MakeCounts(32, 4352)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93"},
    {"Test_a3_topk_1_rejected",
     {{"BS", "4096"}},
     {},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93"},
    {"Test_a3_nonintegral_topk_rejected",
     {{"BS", "255"}},
     {},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93"},
    {"Test_a3_zero_bs_rejected", {{"BS", "0"}}, {}, {}, ge::GRAPH_FAILED, DEFAULT_CCL_BUFFER_SIZE, 8UL, "Ascend910_93"},
    {"Test_a3_experts_1026_rejected",
     {{"epWorldSize", "2"}, {"e", "513"}},
     {{"sendCounts", MakeCounts(1026, 4096)}, {"recvCounts", MakeCounts(1026, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     2UL,
     "Ascend910_93"},
    {"Test_a2_topk_16_rejected", {{"BS", "256"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_aicpu_topk_16_rejected",
     {{"BS", "256"}},
     {},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_BF16,
     "ai_cpu"},
    {"Test_aicpu_experts_1024_rejected",
     {{"e", "128"}},
     {{"sendCounts", MakeCounts(1024, 4096)}, {"recvCounts", MakeCounts(1024, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_BF16,
     "ai_cpu"},
    {"Test_fp16_a3_dense",
     {},
     {},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     1536U,
     ge::DT_FLOAT16},
    {"Test_fp16_a3_one_expert",
     {{"e", "1"}},
     {{"sendCounts", MakeCounts(8, 4096)}, {"recvCounts", MakeCounts(8, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_fp16_a2", {}, {}, {}, ge::GRAPH_SUCCESS, DEFAULT_CCL_BUFFER_SIZE, 8UL, "Ascend910B", 0U, ge::DT_FLOAT16},
    {"Test_fp16_mixed_gmm_weight",
     {},
     {},
     {{1, ge::DT_BF16}},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_fp16_mixed_output",
     {},
     {},
     {{6, ge::DT_BF16}},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_fp16_mixed_shared_input",
     {},
     {},
     {{4, ge::DT_BF16}},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_fp16_mixed_shared_weight",
     {},
     {},
     {{5, ge::DT_BF16}},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_fp16_mixed_shared_output",
     {},
     {},
     {{7, ge::DT_BF16}},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT16},
    {"Test_float32_rejected",
     {},
     {},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U,
     ge::DT_FLOAT},
    {"Test_sample", {}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_ccl_buffer_exact", {}, {}, {}, ge::GRAPH_SUCCESS, SAMPLE_REQUIRED_CCL_BUFFER_SIZE},
    {"Test_ccl_buffer_too_small", {}, {}, {}, ge::GRAPH_FAILED, SAMPLE_REQUIRED_CCL_BUFFER_SIZE - 1ULL},
    {"Test_ccl_buffer_unaligned", {}, {}, {}, ge::GRAPH_FAILED, SAMPLE_REQUIRED_CCL_BUFFER_SIZE + 1ULL},
    {"Test_overlap_a3_dense", {}, {}, {}, ge::GRAPH_SUCCESS, DEFAULT_CCL_BUFFER_SIZE, 8UL, "Ascend910_93", 1536U},
    {"Test_overlap_a2_rank16_rejected",
     {{"epWorldSize", "16"}},
     {{"sendCounts", MakeCounts(64, 4096)}, {"recvCounts", MakeCounts(64, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     16UL},
    {"Test_overlap_a3_one_expert",
     {{"e", "1"}},
     {{"sendCounts", MakeCounts(8, 4096)}, {"recvCounts", MakeCounts(8, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     8UL,
     "Ascend910_93",
     0U},
    {"Test_a3_rank_128",
     {{"epWorldSize", "128"}, {"e", "1"}},
     {{"sendCounts", MakeCounts(128, 4096)}, {"recvCounts", MakeCounts(128, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     128UL,
     "Ascend910_93"},
    {"Test_a3_rank_129_rejected",
     {{"epWorldSize", "129"}, {"e", "1"}},
     {{"sendCounts", MakeCounts(129, 4096)}, {"recvCounts", MakeCounts(129, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     129UL,
     "Ascend910_93"},
    {"Test_a3_rank_256_rejected",
     {{"epWorldSize", "256"}, {"e", "1"}},
     {{"sendCounts", MakeCounts(256, 4096)}, {"recvCounts", MakeCounts(256, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     256UL,
     "Ascend910_93"},
    {"Test_a3_rank_512_rejected",
     {{"epWorldSize", "512"}, {"e", "1"}},
     {{"sendCounts", MakeCounts(512, 4096)}, {"recvCounts", MakeCounts(512, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     512UL,
     "Ascend910_93"},
    {"Test_a3_rank_768_rejected",
     {{"epWorldSize", "768"}, {"e", "1"}},
     {{"sendCounts", MakeCounts(768, 4096)}, {"recvCounts", MakeCounts(768, 4096)}},
     {},
     ge::GRAPH_FAILED,
     DEFAULT_CCL_BUFFER_SIZE,
     768UL,
     "Ascend910_93"},
    {"Test_global_expert_1024_a3",
     {{"epWorldSize", "128"}, {"e", "8"}},
     {{"sendCounts", MakeCounts(1024, 4096)}, {"recvCounts", MakeCounts(1024, 4096)}},
     {},
     ge::GRAPH_SUCCESS,
     DEFAULT_CCL_BUFFER_SIZE,
     128UL,
     "Ascend910_93",
     1536U},
    {"Test_global_expert_too_large",
     {{"epWorldSize", "128"}, {"e", "9"}},
     {{"sendCounts", MakeCounts(1152, 4096)}, {"recvCounts", MakeCounts(1152, 4096)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_BSK_1", {{"BSK", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12889"}, {"mmWeightDim0", "12889"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"epWorldSize", "4"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_size", {{"epWorldSize", "16"}, {"e", "32"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmmWeightDim1", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mmWeightDim0", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"A", "16386"}},
     {{"sendCounts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_0",
     {{"BSK", "16386"}, {"BS", "8193"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_1",
     {{"BSK", "16386"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(GroupedMatMulAlltoAllvMte, GroupedMatMulAlltoAllvMteArch22TilingTest,
                         testing::ValuesIn(g_testParams),
                         [](const testing::TestParamInfo<GroupedMatMulAlltoAllvMteArch22TilingTest::ParamType> &info) {
                             return info.param.testName;
                         });

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim1)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim2)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{7168, 4096}, {7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim3)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim4)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2048, 7168}, {2, 2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim5)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 7168, 6464}, {2, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, Dim6)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2048, 64}, {2, 2048, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvMteArch22TilingTest, TransMmWeightInvalid)
{
    struct GroupedMatMulAlltoAllvMteCompileInfo {};
    GroupedMatMulAlltoAllvMteCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData);
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
} // namespace GroupedMatMulAlltoAllvMteUT
