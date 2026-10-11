/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#include <iostream>

#include <gtest/gtest.h>
#include "../../../op_host/moe_token_unpermute_with_routing_map_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include <nlohmann/json.hpp>

using namespace std;
using namespace ge;

namespace {
void InitParsePlatformInfo(fe::PlatFormInfos& platformInfo)
{
    platformInfo.Init();
    const char* compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileJson);
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    socInfos["ai_core_cnt"] = to_string(compileInfoJson["hardware_info"]["CORE_NUM"].get<uint32_t>());
    socInfos["l2_size"] = to_string(compileInfoJson["hardware_info"]["L2_SIZE"].get<uint32_t>());
    socInfos["core_type_list"] = "AICore";
    aicoreSpec["ub_size"] = to_string(compileInfoJson["hardware_info"]["UB_SIZE"].get<uint32_t>());
    aicoreSpec["l1_size"] = to_string(compileInfoJson["hardware_info"]["L1_SIZE"].get<uint32_t>());
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("version", versions);
}
} // namespace

class MoeTokenUnpermuteWithRoutingMapGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_bf16_tilingkey_1)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo, socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "30 1 0 64 64 30 30 10 1 0 1 0 64 1 0 64 0 1 16 64 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_bf16_tilingkey_11)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo, socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 11;
    string expectTilingData = "30 0 0 64 64 30 30 10 0 0 1 0 64 1 8 64 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_none_bf16_tilingkey_0)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo, socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 0;
    string expectTilingData = "30 1 0 0 64 30 30 10 0 0 1 0 64 1 0 64 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tilingkey_101)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo, socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 101;
    string expectTilingData = "30 1 0 8 8 30 30 10 1 0 1 0 16 1 0 8 0 1 16 32 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tilingkey_111)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo, socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 111;
    string expectTilingData = "30 0 3 8 8 30 30 10 0 0 1 0 16 1 8 8 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, moe_token_unpermute_with_routing_map_grad_tiling_data_accessor)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradTilingData tilingData;
    tilingData.set_tokensNum(1);
    tilingData.set_topK(2);
    tilingData.set_capacity(3);
    tilingData.set_numExpert(4);
    tilingData.set_hiddenSize(5);
    tilingData.set_numOutTokens(6);
    tilingData.set_formerCoreNum(7);
    tilingData.set_tailCoreNum(8);
    tilingData.set_tokenNumEachCore(9);
    tilingData.set_tokenNumTailCore(10);
    tilingData.set_rowIdMapEachCore(11);
    tilingData.set_rowIdMapTailCore(12);
    tilingData.set_hiddenSizeAlign(13);
    tilingData.set_hiddenSizeLoopTimes(14);
    tilingData.set_hiddenSizeLoopTimesAlign(15);
    tilingData.set_hiddenSizeTail(16);
    tilingData.set_inputReserveNum(17);
    tilingData.set_indicesReserveNum(18);
    tilingData.set_indicesReserveNumAlign(19);
    tilingData.set_numExpertAlign(20);
    tilingData.set_totalUbSize(21);

    EXPECT_EQ(tilingData.get_tokensNum(), 1);
    EXPECT_EQ(tilingData.get_topK(), 2);
    EXPECT_EQ(tilingData.get_capacity(), 3);
    EXPECT_EQ(tilingData.get_numExpert(), 4);
    EXPECT_EQ(tilingData.get_hiddenSize(), 5);
    EXPECT_EQ(tilingData.get_numOutTokens(), 6);
    EXPECT_EQ(tilingData.get_formerCoreNum(), 7);
    EXPECT_EQ(tilingData.get_tailCoreNum(), 8);
    EXPECT_EQ(tilingData.get_tokenNumEachCore(), 9);
    EXPECT_EQ(tilingData.get_tokenNumTailCore(), 10);
    EXPECT_EQ(tilingData.get_rowIdMapEachCore(), 11);
    EXPECT_EQ(tilingData.get_rowIdMapTailCore(), 12);
    EXPECT_EQ(tilingData.get_hiddenSizeAlign(), 13);
    EXPECT_EQ(tilingData.get_hiddenSizeLoopTimes(), 14);
    EXPECT_EQ(tilingData.get_hiddenSizeLoopTimesAlign(), 15);
    EXPECT_EQ(tilingData.get_hiddenSizeTail(), 16);
    EXPECT_EQ(tilingData.get_inputReserveNum(), 17);
    EXPECT_EQ(tilingData.get_indicesReserveNum(), 18);
    EXPECT_EQ(tilingData.get_indicesReserveNumAlign(), 19);
    EXPECT_EQ(tilingData.get_numExpertAlign(), 20);
    EXPECT_EQ(tilingData.get_totalUbSize(), 21);
    EXPECT_GT(tilingData.GetDataSize(), 0);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_none_small_ub_hidden_split)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 512}, {30, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 512}, {30, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 512}, {30, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo, "Ascend910B", 40, 512);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_pad_true_small_ub_hidden_split)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo, "Ascend910B", 40, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 11);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_none_invalid_unpermuted_dtype)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_pad_true_invalid_unpermuted_dtype)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_int64_unpermuted_dtype)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_double_unpermuted_dtype)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_parse)
{
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char* compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenUnpermuteWithRoutingMapGrad"))
                      .OpName(ge::AscendString("MoeTokenUnpermuteWithRoutingMapGrad"))
                      .IOInstanceNum({1, 1, 1, 1, 1, 1}, {1, 1})
                      .InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto* parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenUnpermuteWithRoutingMapGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
