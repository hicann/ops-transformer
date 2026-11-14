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
 * \file test_moe_ep_dispatch_tiling.cpp
 * \brief MoeEpDispatch host tiling UT
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../../../common/op_api/moe_ep_comm_alg.h"

namespace MoeEpDispatchUT {

static const std::string OP_NAME = "MoeEpDispatch";
static constexpr uint64_t SKIP_KEY = UINT64_MAX;

struct MoeEpDispatchTestParam {
    std::string caseName;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t numExperts;
    int64_t nmt;
    int64_t cclBufferSize;
    int64_t expertAlignment;
    bool doCpuSync;
    int64_t topoType;
    int64_t rankNumPerServer;
    ge::DataType xDtype;
    bool withTopkWeights;
    bool withCachedSlot;
    int64_t numTokens;
    int64_t hidden;
    int64_t topK;
    ge::graphStatus status;
};

inline std::ostream& operator<<(std::ostream& os, const MoeEpDispatchTestParam& param)
{
    return os << param.caseName;
}

static MoeEpDispatchTestParam g_testCases[] = {
    {"success_direct_bf16", 2, 0, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2,
     ge::GRAPH_SUCCESS},
    {"success_with_weights", 2, 0, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_FLOAT16, true, false, 4, 128, 2,
     ge::GRAPH_SUCCESS},
    {"success_hybrid", 8, 0, 16, 8, 64 * 1024 * 1024, 1, true, 1, 4, ge::DT_BF16, false, false, 4, 256, 2,
     ge::GRAPH_SUCCESS},
    {"success_cached_direct", 2, 0, 4, 8, 16 * 1024 * 1024, 1, false, 0, 1, ge::DT_BF16, false, true, 4, 128, 2,
     ge::GRAPH_SUCCESS},
    {"fail_ep_world", 1, 0, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_ep_rank", 2, 2, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_num_experts", 2, 0, 5, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_nmt", 2, 0, 4, 0, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2, ge::GRAPH_FAILED},
    {"fail_ccl", 2, 0, 4, 8, 0, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2, ge::GRAPH_FAILED},
    {"fail_expert_alignment", 2, 0, 4, 8, 16 * 1024 * 1024, 2, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_topo", 2, 0, 4, 8, 16 * 1024 * 1024, 1, true, 3, 1, ge::DT_BF16, false, false, 4, 128, 2, ge::GRAPH_FAILED},
    {"fail_x_dtype", 2, 0, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_FLOAT, false, false, 4, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_tokens_gt_nmt", 2, 0, 4, 4, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 8, 128, 2,
     ge::GRAPH_FAILED},
    {"fail_hidden", 2, 0, 4, 8, 16 * 1024 * 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 9000, 2,
     ge::GRAPH_FAILED},
    {"fail_ccl_small", 2, 0, 4, 8, 1024, 1, true, 0, 1, ge::DT_BF16, false, false, 4, 128, 2, ge::GRAPH_FAILED},
};

class MoeEpDispatchArch35TilingTest : public testing::TestWithParam<MoeEpDispatchTestParam> {};

static struct MoeEpDispatchCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeEpDispatchTestParam& param,
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

    const int64_t ctxElements = contextElements > 0 ? contextElements : 2052;
    gert::StorageShape contextShape = {{ctxElements}, {ctxElements}};
    gert::StorageShape xShape = {{tokens, hidden}, {tokens, hidden}};
    gert::StorageShape topkShape = {{tokens, topK}, {tokens, topK}};

    std::vector<TD> inputs = {
        {contextShape, ge::DT_INT32, ge::FORMAT_ND},
        {xShape, param.xDtype, ge::FORMAT_ND},
        {topkShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    if (param.withTopkWeights) {
        inputs.push_back({topkShape, ge::DT_FLOAT, ge::FORMAT_ND});
    } else {
        inputs.push_back(ABSENT);
    }
    // scales
    inputs.push_back(ABSENT);
    // cached_slot (+ optional cached routes absent for direct cached)
    if (param.withCachedSlot) {
        gert::StorageShape cachedSlot = {{ep, tokens + 1}, {ep, tokens + 1}};
        inputs.push_back({cachedSlot, ge::DT_INT32, ge::FORMAT_ND});
    } else {
        inputs.push_back(ABSENT);
    }
    inputs.push_back(ABSENT);
    inputs.push_back(ABSENT);
    inputs.push_back(ABSENT);

    bool hybrid = (param.topoType == 1 && param.rankNumPerServer > 0 && ep % param.rankNumPerServer == 0 &&
                   (ep / param.rankNumPerServer) > 1);
    gert::StorageShape dstSlotShape = hybrid ? gert::StorageShape{{tokens, topK}, {tokens, topK}} :
                                               gert::StorageShape{{ep, tokens + 1}, {ep, tokens + 1}};

    std::vector<TD> outputs = {
        {{{ep}, {ep}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{numLocal}, {numLocal}}, ge::DT_INT64, ge::FORMAT_ND},
        {dstSlotShape, ge::DT_INT32, ge::FORMAT_ND},
        {{{tokens}, {tokens}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{tokens, topK}, {tokens, topK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{tokens, topK}, {tokens, topK}}, ge::DT_INT32, ge::FORMAT_ND},
    };

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankId)},
        {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numExperts)},
        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.nmt)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.cclBufferSize)},
        {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expertAlignment)},
        {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(param.doCpuSync)},
        {"host_pinned_counter_addr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.topoType)},
        {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankNumPerServer)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(commAlg)},
    };

    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, &compileInfo, "3510");
}

TEST_P(MoeEpDispatchArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    ExecuteTestCase(BuildTilingContextPara(param), param.status, SKIP_KEY, "", {});
}

INSTANTIATE_TEST_CASE_P(MoeEpDispatchTilingUT, MoeEpDispatchArch35TilingTest, testing::ValuesIn(g_testCases));

TEST(MoeEpDispatchArch35RoutingTest, ClosOnlyUsesSharedTemplateStartingAt16Ranks)
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
        param.doCpuSync = false;
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

TEST(MoeEpDispatchArch35RoutingTest, ClosOnlyRejectsHybridTopology)
{
    auto param = g_testCases[0];
    param.epWorldSize = 8;
    param.numExperts = 16;
    param.topoType = 1;
    param.rankNumPerServer = 4;
    ExecuteTestCase(BuildTilingContextPara(param, "clos_only", 2116), ge::GRAPH_FAILED, SKIP_KEY, "", {});
}

TEST(MoeEpCommAlgTest, AcceptsOnlySupportedNames)
{
    EXPECT_TRUE(Mc2Aclnn::IsValidMoeEpCommAlg("auto"));
    EXPECT_TRUE(Mc2Aclnn::IsValidMoeEpCommAlg("clos_only"));
    EXPECT_TRUE(Mc2Aclnn::IsValidMoeEpCommAlg(nullptr)); // Omitted optional attribute uses auto.
    EXPECT_FALSE(Mc2Aclnn::IsValidMoeEpCommAlg(""));
    EXPECT_FALSE(Mc2Aclnn::IsValidMoeEpCommAlg("shared_jetty"));
    EXPECT_FALSE(Mc2Aclnn::IsValidMoeEpCommAlg("fullmesh"));
}

TEST(MoeEpCommAlgTest, PortPolicyAndTemplateAreResolvedSeparately)
{
    for (int64_t ep : {2, 8, 15, 16, 32, 64}) {
        const auto expectedMode =
            ep >= 16 ? Mc2Tiling::MOE_EP_COMM_ALG_SHARED_JETTY : Mc2Tiling::MOE_EP_COMM_ALG_CLOS_ONLY;
        EXPECT_EQ(Mc2Aclnn::ParseMoeEpCommAlg("clos_only", ep), expectedMode);
        EXPECT_EQ(Mc2Aclnn::ParseMoeEpCommAlg("auto", ep), Mc2Tiling::MOE_EP_COMM_ALG_AUTO);
        EXPECT_EQ(Mc2Aclnn::ParseMoeEpCommAlg(nullptr, ep), Mc2Tiling::MOE_EP_COMM_ALG_AUTO);
    }
    EXPECT_NE(Mc2Tiling::MOE_EP_COMM_ALG_SHARED_JETTY, Mc2Tiling::MOE_EP_COMM_ALG_CLOS_ONLY);
}

} // namespace MoeEpDispatchUT
