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
#include <limits>
#include <map>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/moe_re_routing_tiling.h"

using namespace std;
using namespace ge;

namespace {
constexpr uint32_t ASCEND910B_CORE_NUM = 64;
constexpr uint32_t ASCEND910B_L2_SIZE = 33554432;
constexpr uint32_t ASCEND910B_UB_SIZE = 262144;
constexpr uint32_t ASCEND910B_L1_SIZE = 524288;

void InitParsePlatformInfo(fe::PlatFormInfos &platformInfo)
{
    platformInfo.Init();
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    socInfos["ai_core_cnt"] = to_string(ASCEND910B_CORE_NUM);
    socInfos["l2_size"] = to_string(ASCEND910B_L2_SIZE);
    socInfos["core_type_list"] = "AICore";
    aicoreSpec["ub_size"] = to_string(ASCEND910B_UB_SIZE);
    aicoreSpec["l1_size"] = to_string(ASCEND910B_L1_SIZE);
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("version", versions);
}

void TouchAllMoeReRoutingTilingDataFields()
{
    optiling::MoeReRoutingTilingData membaseData;
    membaseData.set_coreNum(16);
    membaseData.set_ubFactor(8192);
    membaseData.set_tokensNum(64);
    membaseData.set_tokensSize(7168);
    membaseData.set_rankNum(4);
    membaseData.set_expertNumPerRank(16);
    membaseData.set_hasScale(1);
    EXPECT_EQ(membaseData.get_coreNum(), 16);
    EXPECT_EQ(membaseData.get_hasScale(), 1);

    optiling::MoeReRoutingReTilingData reData;
    reData.set_tokenSize(7168);
    reData.set_scaleSize(4);
    reData.set_rankNum(16);
    reData.set_expertNum(16);
    reData.set_coreNum(16);
    reData.set_blockNumR(4);
    reData.set_blockFactorR(4);
    reData.set_blockNumE(4);
    reData.set_blockFactorE(4);
    reData.set_ubFactor(130048);
    reData.set_idxType(0);
    reData.set_tokenSizeOrigin(0);
    EXPECT_EQ(reData.get_tokenSize(), 7168);
    EXPECT_EQ(reData.get_scaleSize(), 4);

    optiling::MoeReRoutingRTilingData rData;
    rData.set_tokenSize(7168);
    rData.set_scaleSize(1);
    rData.set_rankNum(16);
    rData.set_expertNum(16);
    rData.set_coreNum(16);
    rData.set_blockFactor(1);
    rData.set_ubFactor(130048);
    rData.set_idxType(0);
    rData.set_tokenSizeOrigin(0);
    EXPECT_EQ(rData.get_tokenSize(), 7168);
    EXPECT_EQ(rData.get_scaleSize(), 1);
}
} // namespace

class MoeReRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeReRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeReRoutingTiling TearDown" << std::endl;
    }
};

static std::map<std::string, std::string> short_soc_version = {{"Short_SoC_version", "Ascend950"}};
static std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_000) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 100100;
    string expectTilingData = "16 8 256 7168 16 16 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_block_1) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{1, 7168}, {1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{1, 7168}, {1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 100100;
    string expectTilingData = "16 8 1 7168 16 16 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_001) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 100000;
    string expectTilingData = "16 17 64 7168 16 16 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_tiling_000) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo,"Ascend950");
    uint64_t expectTilingKey = 210100;
    string expectTilingData = "7168 1 16 16 16 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_tiling_block_1) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{1, 7168}, {1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{1, 7168}, {1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo,"Ascend950");
    uint64_t expectTilingKey = 210100;
    string expectTilingData = "7168 1 16 16 16 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_tiling_001) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
                                                    {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo,"Ascend950");
    uint64_t expectTilingKey = 200100;
    string expectTilingData = "7168 1 4 4 16 4 1 4 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_tiling_002) {
//     optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
//     gert::TilingContextPara tilingContextPara("MoeReRouting",
//                                                 {
//                                                     {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
//                                                     {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 },
//                                                 {
//                                                     {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                     {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                 },
//                                                 {
//                                                     {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
//                                                     {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//                                                 },
//                                                 &compileInfo,"Ascend950");
//     std::vector<size_t> expectWorkspaces = {};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", expectWorkspaces);
// }

// TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_tiling_003) {
//     optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
//     gert::TilingContextPara tilingContextPara("MoeReRouting",
//                                                 {
//                                                     {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
//                                                     {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 },
//                                                 {
//                                                     {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                     {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                 },
//                                                 {
//                                                     {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
//                                                     {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//                                                 },
//                                                 &compileInfo,"Ascend950");
//     std::vector<size_t> expectWorkspaces = {};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", expectWorkspaces);
// }
// HIF8 + float32 scale test cases
TEST_F(MoeReRoutingTiling, moe_re_routing_hif8_fp32_scale_s32) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo,"Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "7168 4 16 16 16 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_hif8_fp32_scale_s64) {
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo,"Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "7168 4 16 16 16 1 129024 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP4_E2M1 + float32 scale test cases
TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e2m1_fp32_scale_s32)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 4 16 16 16 1 130048 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e2m1_fp32_scale_s64)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 4 16 16 16 1 129024 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP4_E2M1 + float8_e8m0 scale test cases
TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e2m1_e8m0_scale_s32)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 1 16 16 16 1 130048 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e2m1_e8m0_scale_s64)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 1 16 16 16 1 129024 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP4_E1M2 + float32 scale test cases
TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e1m2_fp32_scale_s32)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 4 16 16 16 1 130048 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e1m2_fp32_scale_s64)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 4 16 16 16 1 129024 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP4_E1M2 + float8_e8m0 scale test cases
TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e1m2_e8m0_scale_s32)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 1 16 16 16 1 130048 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_fp4_e1m2_e8m0_scale_s64)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeReRouting",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "3584 1 16 16 16 1 129024 0 7168 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_data_all_fields)
{
    TouchAllMoeReRoutingTilingDataFields();
}

TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_parse_success)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {0, 0};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char *compileJson =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeReRouting"))
                      .OpName(ge::AscendString("MoeReRouting"))
                      .IOInstanceNum({1, 1, 1}, {1, 1, 1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto opImpl = spaceRegistry->GetOpImpl("MoeReRouting");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.coreNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_without_scale)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210000;
    string expectTilingData = "7168 0 16 16 16 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_membase_without_scale)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 100100;
    string expectTilingData = "16 8 256 7168 16 16 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_fp8_e5m2_3d_scale)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{64, 4, 2}, {64, 4, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
                                                    {{{64, 4, 2}, {64, 4, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                    {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 210200;
    string expectTilingData = "7168 8 16 16 16 1 130048 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeReRoutingTiling, moe_re_routing_regbase_zero_tokens)
{
    optiling::MoeReRoutingCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{0, 7168}, {0, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{0, 7168}, {0, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max(), "", {1024 * 1024 * 16});
}