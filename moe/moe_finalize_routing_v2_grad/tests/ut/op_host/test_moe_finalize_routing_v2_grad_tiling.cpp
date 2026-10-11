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
#include "../../../op_host/moe_finalize_routing_v2_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

namespace {
void SetupParsePlatformInfo(fe::PlatFormInfos* platformInfo)
{
    platformInfo->Init();
    map<string, string> socInfos = {
        {"ai_core_cnt", "64"},
        {"l2_size", "33554432"},
        {"core_type_list", "AICore"},
    };
    map<string, string> aicoreSpec = {
        {"ub_size", "65536"},    {"l0_a_size", "65536"}, {"l0_b_size", "65536"},
        {"l0_c_size", "131072"}, {"l1_size", "524288"},  {"bt_size", "0"},
    };
    map<string, string> intrinsics = {
        {"Intrinsic_data_move_l12ub", "float16"},
        {"Intrinsic_data_move_l0c2ub", "float16"},
    };
    map<string, string> versions = {
        {"NpuArch", "2201"},
        {"Short_SoC_version", "Ascend910B"},
    };
    platformInfo->SetPlatformRes("SoCInfo", socInfos);
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");
    platformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo->SetPlatformRes("version", versions);
}
} // namespace

class MoeFinalizeRoutingV2GradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradTiling TearDown" << std::endl;
    }
};

// ----------------------------------------------------------------------------------------------------------

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_10001_001)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {
            // input info
            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        // output info
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 10001;                                  // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 8 5 16376 0 0 10001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_10002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {
            // input info
            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
            {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 10002; // tilngkey
    string expectTilingData =
        "5 0 5 5 0 5 0 1 262144 5 16376 16 128 10002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20001_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 8}, {8, 8}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20001; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 8 15 5448 0 0 20001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262144 15 5448 48 640 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_002)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262144 15 5448 48 640 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_003)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {
            // input info
            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
            {grad_y_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expanded_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {scales_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            //{expert_idx_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262144 15 6528 40 1024 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_10001_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 8}, {5, 8}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {
            // input info
            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
            {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 10001;                                  // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 8 5 16376 0 0 10001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_10002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {
            // input info
            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
            {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 10002; // tilngkey
    string expectTilingData =
        "5 0 5 5 0 5 0 1 262144 5 16376 16 128 10002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20021_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 8}, {8, 8}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20001; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 8 15 4088 0 0 20001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262000}, {5, 262000}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262000}, {8, 262000}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262000 15 4088 64 368 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_002)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262000}, {5, 262000}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262000}, {8, 262000}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(26)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20022; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_003)
{
    // input shape
    gert::StorageShape grad_y_shape = {{18, 12849}, {18, 12849}};
    gert::StorageShape expanded_row_idx_shape = {{90}, {90}};
    gert::StorageShape expanded_x_shape = {{20, 58, 12849}, {20, 58, 12849}};
    gert::StorageShape scales_shape = {{18, 5}, {18, 5}};
    gert::StorageShape expert_idx_shape = {{18, 5}, {18, 5}};
    gert::StorageShape bias_shape = {{20, 12849}, {20, 12849}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{20, 12849}, {20, 12849}};
    gert::StorageShape grad_scales_shape = {{18, 5}, {18, 5}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(58)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_004)
{
    // input shape
    gert::StorageShape grad_y_shape = {{18, 12849}, {18, 12849}};
    gert::StorageShape expanded_row_idx_shape = {{90}, {90}};
    gert::StorageShape expanded_x_shape = {{20, 58, 12849}, {20, 58, 12849}};
    gert::StorageShape scales_shape = {{18, 5}, {18, 5}};
    gert::StorageShape expert_idx_shape = {{18, 5}, {18, 5}};
    gert::StorageShape bias_shape = {{20, 12849}, {20, 12849}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{20, 12849}, {20, 12849}};
    gert::StorageShape grad_scales_shape = {{18, 5}, {18, 5}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad", // op_name
        {                           // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_row_idx_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
         {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {
            {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // attr
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(58)},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData =
        "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_tiling_reset_twice)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10001, "", expectWorkspaces);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, TilingParse_Success)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {0, 0};
    fe::PlatFormInfos platformInfo;
    SetupParsePlatformInfo(&platformInfo);

    const char* compileInfoString =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, )"
        R"("Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, )"
        R"("Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 65536, "L2_SIZE": 33554432, )"
        R"("L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, )"
        R"("CORE_NUM": 64, "socVersion": "Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType("MoeFinalizeRoutingV2Grad")
                      .OpName("MoeFinalizeRoutingV2Grad")
                      .IONum(2, 2)
                      .CompiledJson(compileInfoString)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    SetupParsePlatformInfo(parseContext->GetPlatformInfo());

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2Grad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_invalid_expanded_x_last_dim)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{15, 16}, {15, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_invalid_expanded_x_dim0)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{14, 8}, {14, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_invalid_grad_y_row_idx_without_scales)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_invalid_scales_grad_y_dim0)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6, 3}, {6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2Grad_invalid_bias_expert_num_droppad)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRoutingV2Grad",
        {
            {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{20, 58, 8}, {20, 58, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7, 8}, {7, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{20, 8}, {20, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(58)},
        },
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20001, "", expectWorkspaces);
}
