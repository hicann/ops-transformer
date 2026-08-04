/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_gating_top_k_backward_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/moe_gating_top_k_backward_tiling.h"

using namespace std;

class MoeGatingTopKBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKBackwardTiling TearDown" << std::endl;
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

// ============================================
// Success cases
// ============================================

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 32 32 47 1 32 1 32 2048 256 8 4 0 1 2178868143328329728 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_02)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{4096, 384}, {4096, 384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4096, 384}, {4096, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0.1f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 64 64 35 2 29 2 29 4096 384 8 2 0 1 2178868143299808461 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_03)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{4096, 192}, {4096, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 10}, {4096, 10}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 10}, {4096, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4096, 192}, {4096, 192}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(2.5f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-10f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 64 64 63 2 1 2 1 4096 192 10 2 0 1 3376546329611206656 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_regbase_succ_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 192}, {2048, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 10}, {2048, 10}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 10}, {2048, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 192}, {2048, 192}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_regbase_fail_norm_type_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 192}, {2048, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 10}, {2048, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 10}, {2048, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 192}, {2048, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10000, "", {});
}

// ============================================
// Fail cases
// ============================================

// Boundary cases 1: empty tensor
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_empty_tensor_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_empty_tensor_02)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_empty_tensor_03)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_empty_tensor_04)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary cases 2: tensor dimension mismatch
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_dim_x_norm_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_dim_grad_y_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_dim_expert_idx_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_x_dim_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary cases 3 : tensor shape mismatch
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_y_x_norm_m_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_expert_idx_grad_y_m_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_expert_idx_grad_y_k_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_x_grad_y_m_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 256}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_x_grad_y_n_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary cases 4: k not in valid range
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_k_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 0}, {16, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 0}, {16, 0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_k_02)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 5: expertNum not in valid range
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_n_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 2049}, {16, 2049}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 2049}, {16, 2049}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 6: normType != 1
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_norm_type_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary cases 7: data type mismatch
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_x_norm_dtype_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_y_dtype_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_expert_idx_dtype_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_grad_x_dtype_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 8: UB space insufficient (maxRows <= 0)
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_ub_space_01)
{
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKBackward",
        {
            {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 2048}, {16, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        },
        &compileInfo, "Ascend910B", 32, 8192, 4096);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
