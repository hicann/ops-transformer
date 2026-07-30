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
#include <cstdint>
#include <gtest/gtest.h>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v2_tiling.h"
#include "../../../op_host/moe_init_routing_v2_tiling_util.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include <nlohmann/json.hpp>
#include <map>

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

class MoeInitRoutingV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV2Tiling TearDown" << std::endl;
    }
};

// 单核+drop
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_01)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 6, 30}, {8, 6, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    uint64_t expectTilingKey = 10011;
    string expectTilingData = "64 8 30 6 6 8 1 0 1 1 48 1 48 48 48 1 48 48 8192 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 0 48 "
                              "0 1 1 1 1 1 1 1 1 30 30 1 0 48 48 1 1 1 1 1 1 1 1 30 30 1 0 ";
    std::vector<size_t> expectWorkspaces = {16779264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+非drop
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_perf)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 30}, {48, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string expectTilingData = "64 8 30 6 6 8 0 0 0 1 48 1 48 48 48 1 48 48 8192 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 0 48 "
                              "0 1 1 1 1 1 1 1 1 30 30 1 0 48 48 1 1 1 1 1 1 1 1 30 30 1 0 ";
    std::vector<size_t> expectWorkspaces = {16779264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核++dropless  11000
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_one_core_dropless)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{80, 3000}, {80, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{80, 60}, {80, 60}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 3000}, {8, 3000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4800, 3000}, {4800, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4800}, {4800}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    uint64_t expectTilingKey = 20000;
    string expectTilingData =
        "64 80 3000 60 6 8 0 1 0 1 4800 1 4800 4800 4800 1 4800 4800 8192 0 2040 64 0 75 75 75 75 75 75 0 0 0 0 0 0 64 "
        "0 75 75 75 75 75 75 1 1 3000 3000 1 0 64 4800 75 75 75 75 75 75 1 1 3000 3000 1 0 ";
    std::vector<size_t> expectWorkspaces = {16931328};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+drop  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_drop)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 3000}, {32, 200, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);

    uint64_t expectTilingKey = 10012;
    string expectTilingData =
        "64 320 3000 56 200 32 1 0 1 4 4480 1 4480 4480 4480 1 4480 4480 8192 0 2040 64 0 280 280 280 280 280 280 0 0 "
        "0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 0 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 0 ";
    std::vector<size_t> expectWorkspaces = {17351168};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+dropless  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_dropless)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);

    uint64_t expectTilingKey = 10002;
    string expectTilingData =
        "64 320 3000 56 200 32 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8192 0 2040 64 0 280 280 280 280 280 280 0 0 "
        "0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 0 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 0 ";
    std::vector<size_t> expectWorkspaces = {17351168};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_expert_id)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_token)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_util_print_shape)
{
    gert::Shape emptyShape;
    EXPECT_EQ(optiling::PrintShape(emptyShape), "[]");

    gert::Shape shape1({8});
    EXPECT_EQ(optiling::PrintShape(shape1), "[8]");

    gert::Shape shape2({8, 30});
    EXPECT_EQ(optiling::PrintShape(shape2), "[8, 30]");
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_util_print_tensor_desc_null)
{
    EXPECT_EQ(optiling::PrintTensorDesc(nullptr, nullptr), "nil ");
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_util_print_tiling_context)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws = reinterpret_cast<gert::ContinuousVector *>(wsHolder.get());
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto faker = gert::TilingContextFaker();

    std::vector<std::unique_ptr<gert::Tensor>> keepAlive;
    std::vector<gert::Tensor *> inputs;
    std::vector<gert::Tensor *> outputs;

    gert::StorageShape inShape0({8, 30}, {8, 30});
    auto inT0 = std::make_unique<gert::Tensor>(
        inShape0, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    inputs.push_back(inT0.get());
    keepAlive.push_back(std::move(inT0));

    gert::StorageShape inShape1({8, 6}, {8, 6});
    auto inT1 = std::make_unique<gert::Tensor>(
        inShape1, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    inputs.push_back(inT1.get());
    keepAlive.push_back(std::move(inT1));

    gert::StorageShape outShape0({8, 6, 30}, {8, 6, 30});
    auto outT0 = std::make_unique<gert::Tensor>(
        outShape0, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    outputs.push_back(outT0.get());
    keepAlive.push_back(std::move(outT0));

    gert::StorageShape outShape1({48}, {48});
    auto outT1 = std::make_unique<gert::Tensor>(
        outShape1, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    outputs.push_back(outT1.get());
    keepAlive.push_back(std::move(outT1));

    auto holder = faker.SetOpType("MoeInitRoutingV2")
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputTensors(inputs)
                      .OutputTensors(outputs)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .TilingData(tilingData.get())
                      .Workspace(ws)
                      .Build();
    auto *ctx = holder.GetContext();
    ASSERT_NE(ctx, nullptr);

    std::string result = optiling::PrintTilingContext(ctx);
    EXPECT_FALSE(result.empty());

    auto *shape0 = ctx->GetInputShape(0);
    auto *desc0 = ctx->GetInputDesc(0);
    ASSERT_NE(shape0, nullptr);
    ASSERT_NE(desc0, nullptr);
    std::string tdResult = optiling::PrintTensorDesc(shape0, desc0);
    EXPECT_FALSE(tdResult.empty());
    EXPECT_NE(tdResult, "nil ");
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_dropless)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{48, 30}, {48, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{47, 30}, {47, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_expert_idx_int64)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 6}, {8, 6}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 30}, {48, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_drop_multicore)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 3000}, {32, 200, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_large_cols_drop_pad)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 20000}, {320, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 20000}, {32, 200, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_310p_dropless)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 32}, {48, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend310P");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_vms_middle_multicore)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{640, 3000}, {640, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 56}, {640, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 3000}, {32, 200, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{35840}, {35840}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_small_ub_large_cols)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 8192};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 20000}, {320, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 20000}, {32, 200, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_310p_small_ub_dropless)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 8192};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 32}, {320, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{17920, 32}, {17920, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend310P");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_zero_total_length)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 0}, {8, 0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 6, 30}, {8, 6, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_zero_cols)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 0}, {8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 0}, {48, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_small_ub_medium_cols)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 8192};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 8000}, {320, 8000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 8000}, {32, 200, 8000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_zero_aiv_multicore)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {0, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 200, 3000}, {32, 200, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

// 小 UB + 大 cols + 小 perCoreRows：Tiling4SrcToDstCapacityCompute L719 分支
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_small_ub_large_cols_small_percore)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 8192};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{64, 20000}, {64, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 10}, {64, 10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 6, 20000}, {8, 6, 20000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_parse_success)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {0, 0};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char *compileJson =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeInitRoutingV2"))
                      .OpName(ge::AscendString("MoeInitRoutingV2"))
                      .IOInstanceNum({1, 1, 1}, {1, 1, 1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .Build();
    auto *parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_scatter_mode)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{64, 16}, {64, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2}, {64, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 16}, {128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_scatter_early_exit)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 16}, {0, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_scatter_small_data)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_910b_large_cols_else_branch)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 131072}, {8, 131072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 1}, {8, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 131072}, {8, 131072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_regbase_scatter_1d_expert_idx)
{
    optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {});
}
