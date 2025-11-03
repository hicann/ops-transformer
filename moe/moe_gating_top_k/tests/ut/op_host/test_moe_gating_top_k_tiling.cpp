/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/moe_gating_top_k_tiling.h"

using namespace std;

class MoeGatingTopKTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_succ_01) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "16 16 1 1 256 1 8 4 8 32 32 1 0 1 0 0 0 1024 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_succ_02) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 1024 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_succ_03) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 64}, {16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 64}, {16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "16 16 1 1 64 1 8 8 8 8 32 0 0 0 0 1 2178868142262976512 256 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_succ_04) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 72}, {16, 72}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{72}, {72}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 72}, {16, 72}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 16 1 1 72 1 8 5 9 8 32 1 0 0 0 1 2178868142262976512 384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_01) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_02) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_03) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_04) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 16 1 1 256 1 8 3 8 32 32 1 0 1 0 0 0 1024 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_05) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_06) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_07) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_08) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 16 1 1 256 1 8 4 8 32 32 0 0 1 0 0 0 1024 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_09) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_10) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_fail_11) {
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopK",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "16 16 1 1 256 1 8 1 1 256 256 1 0 1 0 0 2178868143328329728 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(MoeGatingTopKTiling, moe_gating_top_k_tiling_regbase_succ_01) {
//     optiling::MoeGatingTopKCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara("MoeGatingTopK",
//                                               {
//                                                 {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                               },
//                                               {
//                                                 {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
//                                                 {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
//                                               },
//                                               {
//                                                 {"k",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//                                                 {"k_group",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
//                                                 {"group_count",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//                                                 {"group_select_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
//                                                 {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//                                                 {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
//                                                 {"out_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
//                                                 {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
//                                                 {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(0)},
//                                               },
//                                               &compileInfo);
//     uint64_t expectTilingKey = 10000;
//     string expectTilingData = "16 16 1 1 256 1 8 4 8 32 32 1 0 1 0 0 0 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, expectTilingData, expectWorkspaces);
// }