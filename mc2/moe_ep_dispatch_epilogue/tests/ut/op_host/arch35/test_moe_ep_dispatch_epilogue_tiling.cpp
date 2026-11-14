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
 * \file test_moe_ep_dispatch_epilogue_tiling.cpp
 * \brief MoeEpDispatchEpilogue host tiling UT
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace MoeEpDispatchEpilogueUT {

static const std::string OP_NAME = "MoeEpDispatchEpilogue";
static constexpr uint64_t WIN_ALIGN = 512UL;
static constexpr uint64_t SKIP_KEY = UINT64_MAX;

static uint64_t AlignWin(uint64_t value)
{
    return (value + WIN_ALIGN - 1UL) / WIN_ALIGN * WIN_ALIGN;
}

static int64_t CalcPackedElements(int64_t epWorldSize, int64_t numTokens, int64_t topK, int64_t aAlloc)
{
    uint64_t metadataRankOffsetsOffset = AlignWin(static_cast<uint64_t>(aAlloc) * 5UL * 4UL);
    uint64_t localRecvIndexOffset =
        metadataRankOffsetsOffset + AlignWin((static_cast<uint64_t>(epWorldSize) + 1UL) * 4UL);
    uint64_t slotEndsOffset =
        localRecvIndexOffset + AlignWin(static_cast<uint64_t>(numTokens) * static_cast<uint64_t>(topK) * 4UL);
    uint64_t slotRowIdsOffset = slotEndsOffset + AlignWin(static_cast<uint64_t>(aAlloc) * 4UL);
    return static_cast<int64_t>((slotRowIdsOffset + AlignWin(static_cast<uint64_t>(aAlloc) * 4UL)) / 4UL);
}

struct MoeEpDispatchEpilogueTestParam {
    std::string caseName;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t numExperts;
    int64_t nmt;
    int64_t cclBufferSize;
    bool hasTopkWeights;
    int64_t topoType;
    int64_t rankNumPerServer;
    ge::DataType xDtype;
    bool withCached;
    int64_t numTokens;
    int64_t hidden;
    int64_t topK;
    int64_t aAlloc;
    ge::graphStatus status;
};

inline std::ostream& operator<<(std::ostream& os, const MoeEpDispatchEpilogueTestParam& param)
{
    return os << param.caseName;
}

static MoeEpDispatchEpilogueTestParam g_testCases[] = {
    {"success_with_weights", 2, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8,
     ge::GRAPH_SUCCESS},
    {"success_no_weights", 2, 0, 4, 8, 16 * 1024 * 1024, false, 0, 1, ge::DT_FLOAT16, false, 4, 128, 2, 8,
     ge::GRAPH_SUCCESS},
    {"success_hybrid", 8, 0, 16, 8, 64 * 1024 * 1024, true, 1, 4, ge::DT_BF16, false, 4, 256, 2, 16, ge::GRAPH_SUCCESS},
    {"success_cached", 2, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, true, 4, 128, 2, 8, ge::GRAPH_SUCCESS},
    {"fail_ep_world", 1, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_ep_rank", 2, 2, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_num_experts", 2, 0, 5, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_nmt", 2, 0, 4, 0, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_ccl", 2, 0, 4, 8, 0, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_topo", 2, 0, 4, 8, 16 * 1024 * 1024, true, 3, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_x_dtype", 2, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_FLOAT, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
    {"fail_hidden", 2, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 9000, 2, 8, ge::GRAPH_FAILED},
    {"fail_a_alloc", 2, 0, 4, 8, 16 * 1024 * 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 100, ge::GRAPH_FAILED},
    {"fail_ccl_small", 2, 0, 4, 8, 1024, true, 0, 1, ge::DT_BF16, false, 4, 128, 2, 8, ge::GRAPH_FAILED},
};

class MoeEpDispatchEpilogueArch35TilingTest : public testing::TestWithParam<MoeEpDispatchEpilogueTestParam> {};

static struct MoeEpDispatchEpilogueCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeEpDispatchEpilogueTestParam& param,
                                                      const std::string& commAlg = "auto", int64_t contextElements = 0)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    using TD = gert::TilingContextPara::TensorDescription;
    static const TD ABSENT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

    int64_t ep = param.epWorldSize > 0 ? param.epWorldSize : 2;
    int64_t numLocal = (param.epWorldSize > 0 && param.numExperts % param.epWorldSize == 0) ?
                           (param.numExperts / param.epWorldSize) :
                           2;
    int64_t tokens = param.numTokens;
    int64_t topK = param.topK;
    int64_t hidden = param.hidden;
    int64_t aAlloc = param.aAlloc;
    int64_t packed = CalcPackedElements(ep, tokens, topK, aAlloc > 0 ? aAlloc : 8);

    const int64_t ctxElements = contextElements > 0 ? contextElements : 2052;
    gert::StorageShape contextShape = {{ctxElements}, {ctxElements}};
    gert::StorageShape xShape = {{tokens, hidden}, {tokens, hidden}};
    gert::StorageShape topkShape = {{tokens, topK}, {tokens, topK}};
    gert::StorageShape numRecvRank = {{ep}, {ep}};
    gert::StorageShape numRecvExpert = {{numLocal}, {numLocal}};

    std::vector<TD> inputs = {
        {contextShape, ge::DT_INT32, ge::FORMAT_ND},  {xShape, param.xDtype, ge::FORMAT_ND},
        {topkShape, ge::DT_INT32, ge::FORMAT_ND},     {numRecvRank, ge::DT_INT32, ge::FORMAT_ND},
        {numRecvExpert, ge::DT_INT64, ge::FORMAT_ND},
    };
    if (param.withCached) {
        gert::StorageShape cachedMeta = {{packed}, {packed}};
        inputs.push_back({cachedMeta, ge::DT_INT32, ge::FORMAT_ND});
    } else {
        inputs.push_back(ABSENT);
    }

    gert::StorageShape recvX = {{aAlloc, hidden}, {aAlloc, hidden}};
    gert::StorageShape recvMeta = {{packed}, {packed}};
    std::vector<TD> outputs = {
        {recvX, param.xDtype, ge::FORMAT_ND},
        {recvMeta, ge::DT_INT32, ge::FORMAT_ND},
    };
    if (param.hasTopkWeights) {
        outputs.push_back({{{aAlloc}, {aAlloc}}, ge::DT_FLOAT, ge::FORMAT_ND});
    } else {
        outputs.push_back(ABSENT);
    }
    // recv_scales required in IR but only validated for fp8; provide placeholder
    outputs.push_back(ABSENT);

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
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

    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, &compileInfo, "3510");
}

TEST_P(MoeEpDispatchEpilogueArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    ExecuteTestCase(BuildTilingContextPara(param), param.status, SKIP_KEY, "", {});
}

INSTANTIATE_TEST_CASE_P(MoeEpDispatchEpilogueTilingUT, MoeEpDispatchEpilogueArch35TilingTest,
                        testing::ValuesIn(g_testCases));

TEST(MoeEpDispatchEpilogueArch35RoutingTest, ClosOnlyUsesSharedTemplateStartingAt16Ranks)
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

TEST(MoeEpDispatchEpilogueArch35RoutingTest, ClosOnlyRejectsHybridTopology)
{
    auto param = g_testCases[0];
    param.epWorldSize = 8;
    param.numExperts = 16;
    param.topoType = 1;
    param.rankNumPerServer = 4;
    ExecuteTestCase(BuildTilingContextPara(param, "clos_only", 2116), ge::GRAPH_FAILED, SKIP_KEY, "", {});
}

} // namespace MoeEpDispatchEpilogueUT
