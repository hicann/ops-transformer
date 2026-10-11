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
#include "../../../op_host/moe_token_permute_with_routing_map_grad_tiling.h"
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

class MoeTokenPermuteWithRoutingMapGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_fp32)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_fp16)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_bf16)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_bf16_mix)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "7168 2 1024 7168 1 0 8 0 8 1 0 4 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_non_pad_fp32)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t tokenNum = 16;
    int64_t topk = 2;
    int64_t hidden = 64;
    int64_t numOutTokens = tokenNum * topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numOutTokens, hidden}, {numOutTokens, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_non_pad_fp16)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t tokenNum = 16;
    int64_t topk = 2;
    int64_t hidden = 64;
    int64_t numOutTokens = tokenNum * topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numOutTokens, hidden}, {numOutTokens, hidden}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_non_pad_small_tokens)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t tokenNum = 8;
    int64_t topk = 2;
    int64_t hidden = 128;
    int64_t numOutTokens = tokenNum * topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numOutTokens, hidden}, {numOutTokens, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_non_pad_long_hidden)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t tokenNum = 16;
    int64_t topk = 2;
    int64_t hidden = 32768;
    int64_t numOutTokens = tokenNum * topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numOutTokens, hidden}, {numOutTokens, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_non_pad_small_ub)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t tokenNum = 128;
    int64_t topk = 4;
    int64_t hidden = 8192;
    int64_t numOutTokens = tokenNum * topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numOutTokens, hidden}, {numOutTokens, hidden}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numOutTokens}, {numOutTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
        &compileInfo, "Ascend910B", 64, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_topk_exceed_max)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad",
                                              {
                                                  {{{1024, 64}, {1024, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 1024}, {1, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_drop_pad_single_core)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    int64_t numExperts = 4;
    int64_t tokenNum = 8;
    int64_t capacity = 2;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {
            {{{numExperts * capacity, 64}, {numExperts * capacity, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * capacity}, {numExperts * capacity}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numExperts * capacity}, {numExperts * capacity}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{tokenNum, 64}, {tokenNum, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum, numExperts}, {tokenNum, numExperts}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts)},
          {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum)},
          {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1000);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_parse)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char* compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenPermuteWithRoutingMapGrad"))
                      .OpName(ge::AscendString("MoeTokenPermuteWithRoutingMapGrad"))
                      .IOInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto* parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteWithRoutingMapGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
