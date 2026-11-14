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
 * \file test_moe_ep_combine_epilogue_tiling.cpp
 * \brief MoeEpCombineEpilogue host tiling UT
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace MoeEpCombineEpilogueUT {

static const std::string OP_NAME = "MoeEpCombineEpilogue";
static constexpr uint64_t WIN_ALIGN = 512UL;
static constexpr uint64_t SKIP_KEY = UINT64_MAX;

static uint64_t AlignWin(uint64_t value)
{
    return (value + WIN_ALIGN - 1UL) / WIN_ALIGN * WIN_ALIGN;
}

static int64_t CalcPackedMetadataElements(int64_t epWorldSize, int64_t numTokens, int64_t topK, int64_t recvCapacity)
{
    uint64_t metadataRankOffsetsOffset = AlignWin(static_cast<uint64_t>(recvCapacity) * 5UL * sizeof(int32_t));
    uint64_t localRecvIndexOffset =
        metadataRankOffsetsOffset + AlignWin((static_cast<uint64_t>(epWorldSize) + 1UL) * sizeof(int32_t));
    uint64_t packedBytes = localRecvIndexOffset +
                           AlignWin(static_cast<uint64_t>(numTokens) * static_cast<uint64_t>(topK) * sizeof(int32_t)) +
                           2UL * AlignWin(static_cast<uint64_t>(recvCapacity) * sizeof(int32_t));
    return static_cast<int64_t>(packedBytes / sizeof(int32_t));
}

struct MoeEpCombineEpilogueTestParam {
    std::string caseName;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t numExperts;
    int64_t nmt;
    int64_t cclBufferSize;
    bool hasTopkWeights;
    int64_t topoType;
    int64_t rankNumPerServer;
    ge::DataType contextDtype;
    ge::DataType xDtype;
    ge::DataType topkIdxDtype;
    ge::DataType metadataDtype;
    ge::DataType topkWeightsDtype;
    ge::DataType combinedXDtype;
    std::initializer_list<int64_t> xShape;
    std::initializer_list<int64_t> topkIdxShape;
    bool useWrongMetadataLen;
    ge::graphStatus status;
};

inline std::ostream& operator<<(std::ostream& os, const MoeEpCombineEpilogueTestParam& param)
{
    return os << param.caseName;
}

static MoeEpCombineEpilogueTestParam g_testCases[] = {
    {"success_with_weights",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_SUCCESS},
    {"success_no_weights",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     false,
     0,
     1,
     ge::DT_INT32,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_FLOAT16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_SUCCESS},
    {"success_hybrid",
     8,
     0,
     16,
     4,
     16 * 1024 * 1024,
     true,
     1,
     4,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {16, 256},
     {4, 2},
     false,
     ge::GRAPH_SUCCESS},
    {"fail_ep_world_size",
     1,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_ep_rank_id",
     2,
     2,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_num_experts",
     2,
     0,
     5,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_nmt_zero",
     2,
     0,
     4,
     0,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_ccl_zero",
     2,
     0,
     4,
     4,
     0,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_topo",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     3,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_context_dtype",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_x_dtype",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_combined_dtype_mismatch",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_FLOAT16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_topk_idx_dtype",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT64,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_weights_dtype",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_x_hidden",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 9000},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_x_capacity",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {100, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
    {"fail_metadata_len",
     2,
     0,
     4,
     4,
     4 * 1024 * 1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     true,
     ge::GRAPH_FAILED},
    {"fail_ccl_too_small",
     2,
     0,
     4,
     4,
     1024,
     true,
     0,
     1,
     ge::DT_INT32,
     ge::DT_BF16,
     ge::DT_INT32,
     ge::DT_INT32,
     ge::DT_FLOAT,
     ge::DT_BF16,
     {8, 128},
     {4, 2},
     false,
     ge::GRAPH_FAILED},
};

class MoeEpCombineEpilogueArch35TilingTest : public testing::TestWithParam<MoeEpCombineEpilogueTestParam> {};

static struct MoeEpCombineEpilogueCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeEpCombineEpilogueTestParam& param,
                                                      const std::string& commAlg = "auto", int64_t contextElements = 0)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    using TD = gert::TilingContextPara::TensorDescription;
    static const TD ABSENT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

    const int64_t ctxElements = contextElements > 0 ? contextElements : 2052;
    gert::StorageShape contextShape = {{ctxElements}, {ctxElements}};
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape topkIdxShape = {param.topkIdxShape, param.topkIdxShape};

    int64_t numTokens = 4;
    int64_t topK = 2;
    int64_t recvCapacity = 8;
    auto xIt = param.xShape.begin();
    if (param.xShape.size() >= 1) {
        recvCapacity = *xIt;
    }
    auto tkIt = param.topkIdxShape.begin();
    if (param.topkIdxShape.size() >= 2) {
        numTokens = *tkIt;
        topK = *(tkIt + 1);
    }
    int64_t packed =
        CalcPackedMetadataElements(param.epWorldSize > 0 ? param.epWorldSize : 2, numTokens > 0 ? numTokens : 4,
                                   topK > 0 ? topK : 2, recvCapacity > 0 ? recvCapacity : 8);
    if (param.useWrongMetadataLen) {
        packed += 1;
    }
    gert::StorageShape metadataShape = {{packed}, {packed}};
    gert::StorageShape combinedXShape = {{numTokens > 0 ? numTokens : 4, *(param.xShape.begin() + 1)},
                                         {numTokens > 0 ? numTokens : 4, *(param.xShape.begin() + 1)}};

    std::vector<TD> inputTensorDesc_ = {
        {contextShape, param.contextDtype, ge::FORMAT_ND},
        {xShape, param.xDtype, ge::FORMAT_ND},
        {topkIdxShape, param.topkIdxDtype, ge::FORMAT_ND},
        {metadataShape, param.metadataDtype, ge::FORMAT_ND},
    };
    if (param.hasTopkWeights) {
        gert::StorageShape weightsShape = {{recvCapacity}, {recvCapacity}};
        inputTensorDesc_.push_back({weightsShape, param.topkWeightsDtype, ge::FORMAT_ND});
    } else {
        inputTensorDesc_.push_back(ABSENT);
    }

    std::vector<TD> outputTensorDesc_ = {{combinedXShape, param.combinedXDtype, ge::FORMAT_ND}};
    if (param.hasTopkWeights) {
        gert::StorageShape combinedWShape = {{numTokens, topK}, {numTokens, topK}};
        outputTensorDesc_.push_back({combinedWShape, ge::DT_FLOAT, ge::FORMAT_ND});
    } else {
        outputTensorDesc_.push_back(ABSENT);
    }

    std::vector<gert::TilingContextPara::OpAttr> attrs_ = {
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankId)},
        {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numExperts)},
        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.nmt)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.cclBufferSize)},
        {"has_topk_weights", Ops::Transformer::AnyValue::CreateFrom<bool>(param.hasTopkWeights)},
        {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.topoType)},
        {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankNumPerServer)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(commAlg)},
    };

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo, "3510");
}

TEST_P(MoeEpCombineEpilogueArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    ExecuteTestCase(tilingContextPara, param.status, SKIP_KEY, "", {});
}

INSTANTIATE_TEST_CASE_P(MoeEpCombineEpilogueTilingUT, MoeEpCombineEpilogueArch35TilingTest,
                        testing::ValuesIn(g_testCases));

TEST(MoeEpCombineEpilogueArch35RoutingTest, ClosOnlyUsesSharedTemplateStartingAt16Ranks)
{
    for (int64_t ep : {2, 8, 15, 16, 32, 64}) {
        SCOPED_TRACE(ep);
        auto param = g_testCases[0];
        param.epWorldSize = ep;
        param.epRankId = 0;
        param.numExperts = ep * 2;
        param.rankNumPerServer = ep;
        param.topoType = 0;
        param.cclBufferSize = 64 * 1024 * 1024;
        TilingInfo automatic;
        TilingInfo closOnly;
        ASSERT_TRUE(ExecuteTiling(BuildTilingContextPara(param, "auto", 2052), automatic));
        const bool sharedJetty = ep >= 16;
        ASSERT_TRUE(ExecuteTiling(BuildTilingContextPara(param, "clos_only", sharedJetty ? 2116 : 2052), closOnly));
        if (sharedJetty) {
            EXPECT_NE(closOnly.tilingKey, automatic.tilingKey);
            ExecuteTestCase(BuildTilingContextPara(param, "clos_only", 2052), ge::GRAPH_FAILED, SKIP_KEY, "", {});
        } else {
            EXPECT_EQ(closOnly.tilingKey, automatic.tilingKey);
        }
        ExecuteTestCase(BuildTilingContextPara(param, "", 2052), ge::GRAPH_FAILED, SKIP_KEY, "", {});
    }
}

TEST(MoeEpCombineEpilogueArch35RoutingTest, ClosOnlyRejectsHybridTopology)
{
    auto param = g_testCases[0];
    param.epWorldSize = 8;
    param.numExperts = 16;
    param.topoType = 1;
    param.rankNumPerServer = 4;
    ExecuteTestCase(BuildTilingContextPara(param, "clos_only", 2116), ge::GRAPH_FAILED, SKIP_KEY, "", {});
}

} // namespace MoeEpCombineEpilogueUT
