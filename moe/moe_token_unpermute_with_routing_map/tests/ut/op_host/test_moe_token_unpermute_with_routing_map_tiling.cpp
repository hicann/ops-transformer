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
#include <map>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_unpermute_with_routing_map_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include <nlohmann/json.hpp>

using namespace std;

namespace {
void InitParsePlatformInfo(fe::PlatFormInfos &platformInfo)
{
    platformInfo.Init();
    const char *compileJson =
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

class MoeTokenUnpermuteWithRoutingMapTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_fp32_droppad)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{40968 * 8, 7168}, {40968 * 8, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 265}, {4096, 256}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(true)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4096, 7168})},
        },
        &compileInfo);
    int64_t expectTilingKey = 1000;
    // each_loop = min(num_tokens_each_core, maxNPerLoopForUb)：321 小于 maxNPerLoopForUb，
    // 故 front/tail 的 each_loop 均为 321（修复前 GetAlign 向下对齐为 0，会申请 0 字节 UB 队列）。
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 4096 8 2568 64 1 321 321 321 0 1 321 321 321 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_bf16)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{40968 * 8, 7168}, {40968 * 8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 265}, {4096, 256}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568 * 8}, {2568 * 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4096, 7168})},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData =
        "7168 80 327744 7168 1 0 64 0 64 1 0 64 4 8 4096 1 4096 4096 0 0 0 0 0 4096 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {2 * 16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_fp32_probs_none_restore_shape)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 2}, {4, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{4, 64}, {4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4, 64})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_parse)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenUnpermuteWithRoutingMap"))
                      .OpName(ge::AscendString("MoeTokenUnpermuteWithRoutingMap"))
                      .IOInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .Build();
    auto *parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenUnpermuteWithRoutingMap");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_indices_int64_dtype)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{16, 128}, {16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({8, 128})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_small_ub_hidden_split)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{64, 8192}, {64, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 8192}, {32, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 8192})},
        },
        &compileInfo, "Ascend910B", 64, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_many_experts_masked_select_tail)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 65}, {4, 65}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 65}, {4, 65}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 64}, {4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4, 64})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_droppad_capacity_warning)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{200, 64}, {200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{200}, {200}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{200}, {200}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{200}, {200}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{200}, {200}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(true)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({10, 64})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1000);
}
