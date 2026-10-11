/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeCombineAddRmsNormNameSpaceUT {
struct TestParam {
    string testName{};
    std::vector<std::pair<string, string>> tilingParamsStrPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingDTypesPair{};
    ge::graphStatus status;
};

struct TilingParams {
    int64_t A{64};
    int64_t BSK{192};
    int64_t BS{8};
    int64_t K{8};
    int64_t H{7168};
    int64_t epWorldSize{8};
    int64_t epRankId{0};
    int64_t moeExpertNum{8};
    int64_t tpWorldSize{1};
    int64_t tpRankId{0};
    int64_t expertShardType{0};
    int64_t sharedExpertNum{0};
    int64_t sharedExpertRankNum{0};
    int64_t globalBs{0};
    int64_t outDtype{0};
    int64_t commQuantMode{0};
    int64_t groupListType{0};
    float normEps{1e-6};
    std::string commAlg{""};
    std::string groupEp{"groupEp"};
    std::string groupTp{"groupTp"};
};

class MoeDistributeCombineAddRmsNormArch22TilingTest : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormArch22TilingTest TearDown" << std::endl;
    }
};

std::unordered_map<string, std::function<void(TilingParams& tilingParams, const string& valueStr)>>
    g_tilingParamsStrHandlers = {
        {"BSK", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.BSK = std::stoi(valueStr); }}};

TEST_P(MoeDistributeCombineAddRmsNormArch22TilingTest, CommonTest)
{
    auto testParam = GetParam();
    auto tilingParams = TilingParams{};

    for (auto& kv : testParam.tilingParamsStrPair) {
        if (g_tilingParamsStrHandlers.count(kv.first) != 0) {
            g_tilingParamsStrHandlers[kv.first](tilingParams, kv.second);
        }
    }

    struct MoeDistributeCombineAddRmsNormInfo {};
    MoeDistributeCombineAddRmsNormInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombineAddRmsNorm",
        {{{{tilingParams.A, tilingParams.H}, {tilingParams.A, tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tilingParams.BS, tilingParams.K}, {tilingParams.BS, tilingParams.K}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tilingParams.A * 128}, {tilingParams.A * 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tilingParams.epWorldSize}, {tilingParams.epWorldSize}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tilingParams.BS, tilingParams.K}, {tilingParams.BS, tilingParams.K}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{tilingParams.BS, 1, tilingParams.H}, {tilingParams.BS, 1, tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tilingParams.H}, {tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tilingParams.tpWorldSize}, {tilingParams.tpWorldSize}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tilingParams.BS}, {tilingParams.BS}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{tilingParams.BS, tilingParams.H}, {tilingParams.BS, tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_BF16, ge::FORMAT_ND},
         {{}, ge::DT_BF16, ge::FORMAT_ND},
         {{}, ge::DT_BF16, ge::FORMAT_ND},
         {{}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tilingParams.BS, 1, tilingParams.H}, {tilingParams.BS, 1, tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tilingParams.BS, 1, 1}, {tilingParams.BS, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{tilingParams.BS, 1, tilingParams.H}, {tilingParams.BS, 1, tilingParams.H}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"groupEp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.groupEp)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.moeExpertNum)},
         {"groupTp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.groupTp)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.groupListType)},
         {"commAlg", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.commAlg)},
         {"normEps", Ops::Transformer::AnyValue::CreateFrom<float>(tilingParams.normEps)},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", 20, 196608);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        uint64_t expectTilingKey = 16UL;
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

static TestParam g_testParams[] = {{"Test_sample", {}, {}, ge::GRAPH_SUCCESS}};

INSTANTIATE_TEST_SUITE_P(
    MoeDistributeCombineAddRmsNormArch22TilingTest, MoeDistributeCombineAddRmsNormArch22TilingTest,
    testing::ValuesIn(g_testParams),
    [](const testing::TestParamInfo<MoeDistributeCombineAddRmsNormArch22TilingTest::ParamType>& info) {
        return info.param.testName;
    });

TEST_F(MoeDistributeCombineAddRmsNormArch22TilingTest, TestTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeCombineAddRmsNorm");
    ASSERT_NE(opImpl, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    auto ret = tilingParseFunc(nullptr);
    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

} // namespace MoeDistributeCombineAddRmsNormNameSpaceUT
