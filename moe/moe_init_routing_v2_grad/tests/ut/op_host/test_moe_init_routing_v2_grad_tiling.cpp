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
#include <fstream>
#include <vector>
#include <map>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "../../../op_host/moe_init_routing_v2_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

using namespace std;

namespace {
void GetPlatFormInfos(const char *compileInfoStr, map<string, string> &socInfos, map<string, string> &aicoreSpec,
                      map<string, string> &intrinsics)
{
    string defaultHardwardInfo = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 65536, "L2_SIZE": 33554432,
                          "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    if (compileInfoJson.type() != nlohmann::json::value_t::object) {
        compileInfoJson = nlohmann::json::parse(defaultHardwardInfo.c_str());
    }

    map<string, string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"}, {"l2_size", "L2_SIZE"}};
    for (auto &t : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            auto &objJson = compileInfoJson["hardware_info"][t.second];
            if (objJson.is_number_integer()) {
                socInfos[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            }
        }
    }
    socInfos["core_type_list"] = "AICore";

    map<string, string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"},    {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"}, {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},    {"bt_size", "BT_SIZE"}};
    for (auto &t : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            aicoreSpec[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
        }
    }

    std::string intrinsicsKeys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub"};
    for (const string &key : intrinsicsKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = "float16";
        }
    }
}

void InitTilingParsePlatform(fe::PlatFormInfos &platformInfo, const char *compileJson)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileJson, socInfos, aicoreSpec, intrinsics);
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};

    platformInfo.Init();
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetPlatformRes("version", versions);
}

} // namespace

class MoeInitRoutingV2GradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV2GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV2GradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_01)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{480, 8}, {480, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{480}, {480}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 8}, {80, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000;
    string expectTilingData = "40 80 0 0 8 6 0 40 2 2 1 8 8 2 1 1 32 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_02)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{16, 5120}, {16, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 5120}, {8, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1001;
    string expectTilingData = "40 8 0 0 5120 2 0 8 1 1 1 5120 5120 1 1 0 20480 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_03)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{16, 5120}, {16, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 5120}, {8, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1002;
    string expectTilingData = "40 8 0 0 5120 2 0 8 1 1 1 5120 5120 1 1 0 20480 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_04)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1010;
    string expectTilingData = "40 10 0 0 512 64 40 10 1 1 2 496 16 32 1 5 1984 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_05)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{80}, {80}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1011;
    string expectTilingData = "40 80 0 0 512 1 40 40 2 2 1 512 512 1 1 0 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_06)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1012;
    string expectTilingData = "40 10 0 0 512 64 40 10 1 1 2 496 16 32 1 5 1984 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_07)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1100;
    string expectTilingData = "40 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_08)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1101;
    string expectTilingData = "40 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_09)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    int64_t expectTilingKey = 1102;
    string expectTilingData = "40 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_10)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_11)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_12)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV2Grad",
        {
            {{{40, 512, 1024, 1024}, {40, 512, 1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_13)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_14)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{196608, 8}, {196608, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{196607}, {196607}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_15)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(30)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_16)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 510}, {10, 510}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_17)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// tiling_base.cpp: CheckDtypeValidity / CheckParamsValidity / GetShapeAttrsInfo 失败分支
TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_dtype_mismatch_18)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_row_idx_dtype_19)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_input_dtype_20)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_top_k_zero_21)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_grad_x_dim_22)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512, 1}, {10, 512, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_invalid_row_idx_not_divisible_23)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{641, 512}, {641, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{641}, {641}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{106, 512}, {106, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_parse_succ_24)
{
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 65536, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40,
            "socVersion": "Ascend910B"}})";

    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitTilingParsePlatform(platformInfo, compileJson);

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeInitRoutingV2Grad"))
                      .OpName(ge::AscendString("MoeInitRoutingV2Grad"))
                      .IONum(2, 1)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<const void *>(&platformInfo))
                      .Build();
    auto parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingV2Grad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    EXPECT_EQ(opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext)), ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

// arch35 / regbase path (Ascend950): MoeInitRoutingV2GradRegbase template (priority 40000)
TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_dropless_01)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 400001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_active_02)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 400003;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_drop_pad_03)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{5120}, {5120}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 400002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_fp32_full_load_04)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 512}, {640, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 512}, {10, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 400001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_large_h_split_ub_05)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 8192}, {640, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 400001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// arch35 / regbase SplitH template (priority 20000): topK<=64, N<aivNum, N < H/(512*typeSize)
TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_split_h_dropless_01)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 8192}, {40, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{40}, {40}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 8192}, {5, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 200001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_split_h_active_02)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{5, 8192}, {5, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{40}, {40}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 8192}, {5, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 200003;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_split_h_drop_pad_03)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{4, 4, 32768}, {4, 4, 32768}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 32768}, {16, 32768}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 200002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_split_h_fp32_04)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{24, 16384}, {24, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24}, {24}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 16384}, {3, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 200001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_split_h_subh_partial_ub_05)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 4096};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 8192}, {40, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{40}, {40}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 8192}, {5, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 200001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// arch35 / regbase FullLoad template (priority 30000): SplitH 不适用后由本模板承接
TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_full_load_dropless_01)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{480, 8}, {480, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{480}, {480}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 8}, {80, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 300001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_full_load_active_02)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{320}, {320}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{40, 512}, {40, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 300003;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_full_load_drop_pad_03)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{10, 8, 512}, {10, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 512}, {80, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 300002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_full_load_fp16_h_full_04)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{16, 5120}, {16, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 5120}, {8, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 300001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_regbase_full_load_partial_h_05)
{
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                              {
                                                  {{{640, 2048}, {640, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{640}, {640}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{80, 2048}, {80, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 300001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// 触达 tiling.h 内 MoeInitRoutingV2GradTilingData / MoeV2GradComputeTilingData 字段宏
TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_data_struct_fields)
{
    optiling::MoeV2GradComputeTilingData computeData;
    computeData.set_needCoreNum(1);
    computeData.set_perCoreElements(2);
    computeData.set_lastCoreElements(3);
    computeData.set_elementCopyLoops(4);
    computeData.set_elementPerCopyCols(5);
    computeData.set_elementLastCopyCols(6);
    computeData.set_binaryAddBufferNum(7);
    computeData.set_tmpBufferNum(8);
    computeData.set_exponentOfBinary(9);
    computeData.set_copyBufferSize(10);
    computeData.set_tokensFormer(11);
    computeData.set_perCoreTokensLoop(12);
    computeData.set_perCoreTailTokensFormer(13);
    computeData.set_lastCoreTokensLoop(14);
    computeData.set_lastCoreTailTokensFormer(15);

    optiling::MoeInitRoutingV2GradTilingData tilingData;
    tilingData.set_coreNum(40);
    tilingData.set_n(80);
    tilingData.set_e(10);
    tilingData.set_c(8);
    tilingData.set_cols(512);
    tilingData.set_k(6);
    tilingData.set_activeNum(40);

    optiling::MoeInitRoutingV2GradRegbaseSplitHTilingData splitHData;
    splitHData.set_n(5);
    splitHData.set_kUbFactor(2);
    splitHData.set_k(8);
    splitHData.set_activeNum(5);
    splitHData.set_h(8192);
    splitHData.set_hBlockFactor(256);
    splitHData.set_hUbFactor(256);
    splitHData.set_numBlocks(32);

    optiling::MoeInitRoutingV2GradRegbaseFullLoadTilingData fullLoadData;
    fullLoadData.set_n(80);
    fullLoadData.set_nBlockFactor(2);
    fullLoadData.set_nUbFactor(4);
    fullLoadData.set_k(6);
    fullLoadData.set_activeNum(0);
    fullLoadData.set_h(8);
    fullLoadData.set_hUbFactor(8);
    fullLoadData.set_numBlocks(40);

    optiling::MoeInitRoutingV2GradRegbaseTilingData regbaseData;
    regbaseData.set_n(10);
    regbaseData.set_nBlockFactor(1);
    regbaseData.set_nUbFactor(1);
    regbaseData.set_k(64);
    regbaseData.set_kUbFactor(8);
    regbaseData.set_activeNum(0);
    regbaseData.set_h(512);
    regbaseData.set_hUbFactor(512);
    regbaseData.set_numBlocks(10);

    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {40, 65536};
    EXPECT_EQ(compileInfo.aivNum, 40);
    EXPECT_EQ(compileInfo.ubSize, 65536U);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_parse_fail_aivnum_zero_25)
{
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 65536, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 0,
            "socVersion": "Ascend910B"}})";

    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitTilingParsePlatform(platformInfo, compileJson);

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeInitRoutingV2Grad"))
                      .OpName(ge::AscendString("MoeInitRoutingV2Grad"))
                      .IONum(2, 1)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<const void *>(&platformInfo))
                      .Build();
    auto parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingV2Grad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    EXPECT_EQ(opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext)), ge::GRAPH_FAILED);
    EXPECT_LE(compileInfo.aivNum, 0);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_parse_fail_ubsize_zero_26)
{
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 256, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40,
            "socVersion": "Ascend910B"}})";

    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitTilingParsePlatform(platformInfo, compileJson);

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeInitRoutingV2Grad"))
                      .OpName(ge::AscendString("MoeInitRoutingV2Grad"))
                      .IONum(2, 1)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<const void *>(&platformInfo))
                      .Build();
    auto parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingV2Grad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    EXPECT_EQ(opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext)), ge::GRAPH_FAILED);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_EQ(compileInfo.ubSize, 0U);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_compile_info_default_ctor_27)
{
    optiling::MoeInitRoutingV2GradCompileInfo defaultInfo;
    EXPECT_EQ(defaultInfo.aivNum, 0);
    EXPECT_EQ(defaultInfo.ubSize, 0U);

    auto *heapInfo = new optiling::MoeInitRoutingV2GradCompileInfo();
    EXPECT_EQ(heapInfo->aivNum, 0);
    EXPECT_EQ(heapInfo->ubSize, 0U);
    delete heapInfo;

    optiling::MoeInitRoutingV2GradCompileInfo copyInfo = defaultInfo;
    EXPECT_EQ(copyInfo.aivNum, 0);
    EXPECT_EQ(copyInfo.ubSize, 0U);

    copyInfo.aivNum = 10;
    copyInfo.ubSize = 4096;
    optiling::MoeInitRoutingV2GradCompileInfo assignInfo;
    assignInfo = copyInfo;
    EXPECT_EQ(assignInfo.aivNum, 10);
    EXPECT_EQ(assignInfo.ubSize, 4096U);
}
