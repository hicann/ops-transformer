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
 * \file test_moe_token_unpermute_tiling.cpp
 * \brief
 */
#include <iostream>
#include <limits>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_unpermute_tiling.h"
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

class MoeTokenUnpermuteTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_bf16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{6144, 20480}, {6144, 20480}}, ge::DT_BF16, ge::FORMAT_ND}, {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{6144, 20480}, {6144, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "20480 1 6144 17920 1 2560 96 0 96 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_bf16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_fp16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "5120 1 6144 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_fp16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 3;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_fp32)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo);
    int64_t expectTilingKey = 4;
    string expectTilingData = "5120 1 6144 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_fp32)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 5;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_bf16_fp32)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 25;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_bf16_fp16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 17;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp16_fp32)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 27;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp16_bf16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 11;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp32_fp16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 21;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp32_bf16)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 13;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_small_tokens_num_core)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_small_ub_token_split)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{512, 4096}, {512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{64, 8}, {64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{64, 4096}, {64, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend910B", 64, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_310p_fp16_success)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo, "Ascend310P");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_310p_hidden_not_aligned)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{128, 100}, {128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 100}, {128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo, "Ascend310P");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_310p_bf16_not_support)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermute",
        {{{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo, "Ascend310P");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_probs_dim_mismatch)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_probs_not_2d)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8, 8, 1}, {8, 8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_probs_topk_too_large)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                              {{{{513, 128}, {513, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{513}, {513}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1, 513}, {1, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{1, 128}, {1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_parse)
{
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char* compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenUnpermute"))
                      .OpName(ge::AscendString("MoeTokenUnpermute"))
                      .IOInstanceNum({1, 1, 1}, {1})
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto* parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenUnpermute");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
