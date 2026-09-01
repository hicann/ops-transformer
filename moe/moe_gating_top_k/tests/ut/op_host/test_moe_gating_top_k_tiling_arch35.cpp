/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/moe_gating_top_k_tiling.h"

using namespace std;

class MoeGatingTopKTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKTilingArch35 TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_01)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_02)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_03)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_04)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_05_norm_type_softplus)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_06_renorm_l1)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_07_group_select_mode_max)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_08_fp16)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_09_bf16)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_10_no_bias)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_11_simplified_path_kgroup_eq_groupcount)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_12_simplified_path_groupcount_eq_expertcount)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_13_e_k_fullload_k32_boundary)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 32}, {16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 32}, {16, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_14_e_k_fullload_k33_fallback_to_regbase)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 33}, {16, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 33}, {16, 33}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_15_without_group_kgroup_eq_groupcount_k1)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10005, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_16_without_group_groupcount_eq_expertcount_k1)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10005, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_17_without_group_fp16)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10005, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_01_invalid_k)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_02_invalid_k_exceed_expert)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_03_invalid_k_group)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_04_invalid_group_count_zero)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_05_expert_not_divisible_by_group)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_06_invalid_group_select_mode)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_07_invalid_renorm)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_08_invalid_renorm_value)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_09_invalid_norm_type)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_10_kgroup_exceed_groupcount)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_fail_11_k_exceed_kgroup_times_groupexpert)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
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
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_succ_01_inputids_int32_tid2eid_int64)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10001, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_succ_02_inputids_int32_tid2eid_int32)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10002, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_succ_03_inputids_int64_tid2eid_int64)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10003, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_succ_04_inputids_int64_tid2eid_int32)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10004, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_succ_05_groupcount_eq_expertcount)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10001, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_01_k_exceed_64)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 65}, {16, 65}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{65, 256}, {65, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 65}, {16, 65}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 65}, {16, 65}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_02_tid2eid_dim1_not_eq_k)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_03_inputids_dim0_not_eq_rows)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_04_non_simplified_path)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_05_inputids_invalid_dtype)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_hash_fail_06_tid2eid_invalid_dtype)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 256}, {8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", {});
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_18_softmax_fp16_k8_regbase)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_19_without_group_softmax_fp16_k1_norenorm)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10005, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_succ_20_without_group_softmax_fp16_k1_renorm)
{
    optiling::MoeGatingTopKCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10005, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKTilingArch35, moe_gating_top_k_tiling_arch35_parse_succ_01)
{
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeGatingTopK");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
