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
#include "../../../op_host/moe_token_permute_with_routing_map_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include <nlohmann/json.hpp>

using namespace std;

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

class MoeTokenPermuteWithRoutingMapTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_float)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 8 3 3 0 0 1 6 1 6 6 6 1 6 6 8160 0 0 0 0 0 0 1024 2 2 0 1 0 20 2 2976 5952 17856 "
                              "1 1 1 1 1 0 5952 2 2976 1 1 6 95232 71424 3 2 1 2 2 0 0 0 0 0 2 3 ";
    std::vector<size_t> expectWorkspaces = {16777384};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_multi_core)
{
    int64_t tokenNum = 3;
    int64_t topk = 2;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, tokenNum}, {3, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 3 5 8 2 3 0 0 1 6 1 6 6 6 1 6 6 8160 0 0 0 0 0 0 1024 3 3 0 1 0 20 2 3273 6546 13092 "
                              "1 1 1 1 1 0 6546 2 3273 1 1 6 104736 52384 3 3 1 3 3 0 0 0 0 0 3 3 ";
    std::vector<size_t> expectWorkspaces = {16777408};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_long_h)
{
    int64_t tokenNum = 3;
    int64_t topk = 2;
    int64_t h = 32768;

    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, tokenNum}, {3, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 3;
    string expectTilingData = "1 3 32768 32768 2 3 0 0 1 6 1 6 6 6 1 6 6 8160 0 0 0 0 0 0 1024 3 3 0 1 0 131072 1 1 "
                              "256 512 384 32384 2 1 1 0 256 256 1 1 1 6 129536 2048 3 3 1 3 3 0 0 0 0 0 3 3 ";
    std::vector<size_t> expectWorkspaces = {16777408};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_error)
{
    int64_t tokenNum = 3;
    int64_t topk = 2;
    int64_t h = 32768;

    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 3 5 8 2 0 0 0 1 6 1 6 6 6 1 6 6 8160 0 0 0 0 0 0 1024 3 3 0 1 0 20 2 3273 6546 13092 "
                              "1 1 1 1 1 0 6546 2 3273 1 1 6 104736 13120 3 3 1 3 3 0 0 0 0 0 3 3 ";
    std::vector<size_t> expectWorkspaces = {16777376};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_float16)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_multi_core_large)
{
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t numExperts = 3;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_with_prob)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_many_experts)
{
    int64_t tokenNum = 1;
    int64_t numExperts = 41;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{numExperts, 5}, {numExperts, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts}, {numExperts}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts}, {numExperts}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_topk_out_of_range)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{513, 2}, {513, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1026, 5}, {1026, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1026}, {1026}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1026}, {1026}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1026)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_indices_dim_invalid)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2, 1}, {3, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_drop_and_pad_regbase)
{
    int64_t tokenNum = 2;
    int64_t numExperts = 3;
    int64_t cols = 5;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, cols}, {tokenNum, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{numExperts * 2, cols}, {numExperts * 2, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 2}, {numExperts * 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 2}, {numExperts * 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts * 2)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 19);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_drop_and_pad_regbase_simd)
{
    int64_t tokenNum = 64;
    int64_t numExperts = 8;
    int64_t cols = 7168;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, cols}, {tokenNum, cols}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{numExperts * 8, cols}, {numExperts * 8, cols}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts * 8}, {numExperts * 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts * 8}, {numExperts * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts * 8)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_drop_and_pad_regbase_simt_small_cols)
{
    int64_t tokenNum = 64;
    int64_t numExperts = 8;
    int64_t cols = 5;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, cols}, {tokenNum, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{numExperts * 64, cols}, {numExperts * 64, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 64}, {numExperts * 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 64}, {numExperts * 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts * 64)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 19);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_with_prob_fp16)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_output_cols_mismatch)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_output_token_mismatch)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_output_rank_invalid)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_parse)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {0, 0, 0};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char* compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenPermuteWithRoutingMap"))
                      .OpName(ge::AscendString("MoeTokenPermuteWithRoutingMap"))
                      .IOInstanceNum({1, 1, 1}, {1, 1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto* parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteWithRoutingMap");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_output_probs_rank_invalid)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6, 1}, {6, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_drop_and_pad_regbase_large_prob)
{
    int64_t tokenNum = 128;
    int64_t numExperts = 8;
    int64_t cols = 5;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, cols}, {tokenNum, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{numExperts * 1024, cols}, {numExperts * 1024, cols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 1024}, {numExperts * 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts * 1024}, {numExperts * 1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts * 1024)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 19);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_zero_token)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 0}, {3, 0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_sort_limit_exceeded)
{
    int64_t tokenNum = 32769;
    int64_t topk = 512;
    int64_t numExperts = 3;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_vbs_ceil_align_branch)
{
    int64_t tokenNum = 20000;
    int64_t topk = 5;
    int64_t numExperts = 3;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_drop_and_pad_regbase_xdtype_b32)
{
    int64_t tokenNum = 64;
    int64_t numExperts = 8;
    int64_t cols = 622;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, cols}, {tokenNum, cols}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts, tokenNum}, {numExperts, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{numExperts * 8, cols}, {numExperts * 8, cols}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts * 8}, {numExperts * 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{numExperts * 8}, {numExperts * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numExperts * 8)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_with_prob_int64)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_with_prob_int8)
{
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_small_routing_map_single_core)
{
    int64_t tokenNum = 4;
    int64_t topk = 2;
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithRoutingMap",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, tokenNum}, {2, tokenNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_data_accessor)
{
    optiling::PermuteVBSComputeRMTilingData vbsData;
    vbsData.set_needCoreNum(1);
    vbsData.set_perCoreElements(2);
    vbsData.set_perCoreLoops(3);
    vbsData.set_perCorePerLoopElements(4);
    vbsData.set_perCoreLastLoopElements(5);
    vbsData.set_lastCoreElements(6);
    vbsData.set_lastCoreLoops(7);
    vbsData.set_lastCorePerLoopElements(8);
    vbsData.set_lastCoreLastLoopElements(9);
    vbsData.set_oneLoopMaxElements(10);
    vbsData.set_lastCoreWSindex(11);
    vbsData.set_frontcoreTask(12);
    vbsData.set_tailcoreTask(13);
    vbsData.set_frontCoreNum(14);
    vbsData.set_tailCoreNum(15);

    EXPECT_EQ(vbsData.get_needCoreNum(), 1);
    EXPECT_EQ(vbsData.get_perCoreElements(), 2);
    EXPECT_EQ(vbsData.get_perCoreLoops(), 3);
    EXPECT_EQ(vbsData.get_perCorePerLoopElements(), 4);
    EXPECT_EQ(vbsData.get_perCoreLastLoopElements(), 5);
    EXPECT_EQ(vbsData.get_lastCoreElements(), 6);
    EXPECT_EQ(vbsData.get_lastCoreLoops(), 7);
    EXPECT_EQ(vbsData.get_lastCorePerLoopElements(), 8);
    EXPECT_EQ(vbsData.get_lastCoreLastLoopElements(), 9);
    EXPECT_EQ(vbsData.get_oneLoopMaxElements(), 10);
    EXPECT_EQ(vbsData.get_lastCoreWSindex(), 11);
    EXPECT_EQ(vbsData.get_frontcoreTask(), 12);
    EXPECT_EQ(vbsData.get_tailcoreTask(), 13);
    EXPECT_EQ(vbsData.get_frontCoreNum(), 14);
    EXPECT_EQ(vbsData.get_tailCoreNum(), 15);

    optiling::PermuteVMSMiddleComputeRMTilingData vmsData;
    vmsData.set_needCoreNum(16);
    EXPECT_EQ(vmsData.get_needCoreNum(), 16);

    optiling::PermuteSortOutComputeRMTilingData sortOutData;
    sortOutData.set_oneLoopMaxElements(17);
    EXPECT_EQ(sortOutData.get_oneLoopMaxElements(), 17);

    optiling::IndexCopyComputeRMTilingData indexData;
    indexData.set_needCoreNum(1);
    indexData.set_frontCoreNum(2);
    indexData.set_tailCoreNum(3);
    indexData.set_coreCalcNum(4);
    indexData.set_coreCalcTail(5);
    indexData.set_oneTokenBtypeSize(6);
    indexData.set_onceIndicesTokenMoveTimes(7);
    indexData.set_onceUbTokenNums(8);
    indexData.set_onceIndicesTokenNums(9);
    indexData.set_onceIndices(10);
    indexData.set_oneTokenlastMove(11);
    indexData.set_oneTokenOnceMove(12);
    indexData.set_oneTokenMoveTimes(13);
    indexData.set_frontCoreLoop(14);
    indexData.set_frontCoreLastTokenNums(15);
    indexData.set_tailCoreLoop(16);
    indexData.set_tailCoreLastTokenNums(17);
    indexData.set_tailLastonceIndicesTokenMoveTimes(18);
    indexData.set_tailLastIndicesLastTokenNums(19);
    indexData.set_frontLastonceIndicesTokenMoveTimes(20);
    indexData.set_frontLastIndicesLastTokenNums(21);
    indexData.set_numOutTokens(22);
    indexData.set_tokenUB(23);
    indexData.set_indicesUB(24);

    EXPECT_EQ(indexData.get_needCoreNum(), 1);
    EXPECT_EQ(indexData.get_frontCoreNum(), 2);
    EXPECT_EQ(indexData.get_tailCoreNum(), 3);
    EXPECT_EQ(indexData.get_coreCalcNum(), 4);
    EXPECT_EQ(indexData.get_coreCalcTail(), 5);
    EXPECT_EQ(indexData.get_oneTokenBtypeSize(), 6);
    EXPECT_EQ(indexData.get_onceIndicesTokenMoveTimes(), 7);
    EXPECT_EQ(indexData.get_onceUbTokenNums(), 8);
    EXPECT_EQ(indexData.get_onceIndicesTokenNums(), 9);
    EXPECT_EQ(indexData.get_onceIndices(), 10);
    EXPECT_EQ(indexData.get_oneTokenlastMove(), 11);
    EXPECT_EQ(indexData.get_oneTokenOnceMove(), 12);
    EXPECT_EQ(indexData.get_oneTokenMoveTimes(), 13);
    EXPECT_EQ(indexData.get_frontCoreLoop(), 14);
    EXPECT_EQ(indexData.get_frontCoreLastTokenNums(), 15);
    EXPECT_EQ(indexData.get_tailCoreLoop(), 16);
    EXPECT_EQ(indexData.get_tailCoreLastTokenNums(), 17);
    EXPECT_EQ(indexData.get_tailLastonceIndicesTokenMoveTimes(), 18);
    EXPECT_EQ(indexData.get_tailLastIndicesLastTokenNums(), 19);
    EXPECT_EQ(indexData.get_frontLastonceIndicesTokenMoveTimes(), 20);
    EXPECT_EQ(indexData.get_frontLastIndicesLastTokenNums(), 21);
    EXPECT_EQ(indexData.get_numOutTokens(), 22);
    EXPECT_EQ(indexData.get_tokenUB(), 23);
    EXPECT_EQ(indexData.get_indicesUB(), 24);

    optiling::MaskedSelectRMTilingData maskedData;
    maskedData.set_formerNum(1);
    maskedData.set_formerLength(2);
    maskedData.set_formertileNum(3);
    maskedData.set_formertileLength(4);
    maskedData.set_formerlasttileLength(5);
    maskedData.set_tailNum(6);
    maskedData.set_tailLength(7);
    maskedData.set_tailtileNum(8);
    maskedData.set_tailtileLength(9);
    maskedData.set_taillasttileLength(10);
    maskedData.set_tokenNum(11);
    maskedData.set_needCoreNum(12);
    EXPECT_EQ(maskedData.get_formerNum(), 1);
    EXPECT_EQ(maskedData.get_needCoreNum(), 12);

    optiling::MoeTokenPermuteWithRoutingMapTilingData tilingData;
    tilingData.set_coreNum(64);
    tilingData.set_n(100);
    tilingData.set_cols(200);
    tilingData.set_colsAlign(256);
    tilingData.set_topK(8);
    tilingData.set_expertNum(16);
    tilingData.set_capacity(32);
    tilingData.set_hasProb(1);
    tilingData.indexCopyComputeParamsOp.set_needCoreNum(indexData.get_needCoreNum());
    tilingData.indexCopyComputeParamsOp.set_tokenUB(indexData.get_tokenUB());
    tilingData.vbsComputeParamsOp.set_oneLoopMaxElements(vbsData.get_oneLoopMaxElements());
    tilingData.vmsMiddleComputeParamsOp.set_needCoreNum(vmsData.get_needCoreNum());

    EXPECT_EQ(tilingData.get_coreNum(), 64);
    EXPECT_EQ(tilingData.get_n(), 100);
    EXPECT_EQ(tilingData.get_cols(), 200);
    EXPECT_EQ(tilingData.get_colsAlign(), 256);
    EXPECT_EQ(tilingData.get_topK(), 8);
    EXPECT_EQ(tilingData.get_expertNum(), 16);
    EXPECT_EQ(tilingData.get_capacity(), 32);
    EXPECT_EQ(tilingData.get_hasProb(), 1);
    EXPECT_EQ(tilingData.indexCopyComputeParamsOp.get_needCoreNum(), 1);
    EXPECT_EQ(tilingData.indexCopyComputeParamsOp.get_tokenUB(), 23);
    EXPECT_EQ(tilingData.vbsComputeParamsOp.get_oneLoopMaxElements(), 10);
    EXPECT_EQ(tilingData.vmsMiddleComputeParamsOp.get_needCoreNum(), 16);
    EXPECT_GT(tilingData.GetDataSize(), 0);

    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {};
    compileInfo.aivNum = 40;
    compileInfo.ubSize = 192 * 1024;
    compileInfo.workSpaceSize = 1024;
    EXPECT_EQ(compileInfo.aivNum, 40U);
    EXPECT_EQ(compileInfo.ubSize, 192U * 1024);
    EXPECT_EQ(compileInfo.workSpaceSize, 1024U);
}
