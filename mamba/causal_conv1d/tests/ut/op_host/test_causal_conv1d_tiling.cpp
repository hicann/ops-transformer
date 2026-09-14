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
 * \file test_causal_conv1d_tiling.cpp
 * \brief Unit tests for CausalConv1d tiling logic.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "../../../op_host/causal_conv1d_tiling.cpp"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// ============================================================
// Common helpers
// ============================================================

static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(std::string activationMode = "silu",
                                                              int64_t nullBlockId = 0)
{
    return {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(activationMode)},
        {"null_block_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(nullBlockId)},
    };
}

// ============================================================
// FN Mode (prefill)
// ============================================================

class CausalConv1dFnTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dFnTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dFnTiling TearDown" << std::endl;
    }
};

TEST_F(CausalConv1dFnTiling, fn_3d_fp16_b2_s16_d8192_w4_bias)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p("CausalConv1d",
                              {
                                  {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // weight
                                  {{{2, 3, 8192}, {2, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states
                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},               // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // query_start_loc
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cache_indices
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // initial_state_mode
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // num_accepted_tokens
                              },
                              {
                                  {{{2, 3, 8192}, {2, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states
                                  {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                              },
                              a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(CausalConv1dFnTiling, fn_2d_varlen_fp16_d8192_w4)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p("CausalConv1d",
                              {
                                  {{{32, 8192}, {32, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x (2D varlen)
                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{2, 3, 8192}, {2, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND}, // query_start_loc (batch=2)
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // cache_indices
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // initial_state_mode
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // num_accepted_tokens
                              },
                              {
                                  {{{2, 3, 8192}, {2, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                  {{{32, 8192}, {32, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
                              },
                              a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ============================================================
// Update Mode (decode)
// ============================================================

class CausalConv1dUpdateTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dUpdateTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dUpdateTiling TearDown" << std::endl;
    }
};

TEST_F(CausalConv1dUpdateTiling, upd_3d_fp16_b128_s1_d8192_w4_bias)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p("CausalConv1d",
                              {
                                  {{{128, 1, 8192}, {128, 1, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
                                  {{{128, 3, 8192}, {128, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // query_start_loc
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cache_indices
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // initial_state_mode
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // num_accepted_tokens
                              },
                              {
                                  {{{128, 3, 8192}, {128, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                  {{{128, 1, 8192}, {128, 1, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                              },
                              a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ============================================================
// Error cases
// ============================================================

TEST_F(CausalConv1dUpdateTiling, reject_int32_dtype)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p("CausalConv1d",
                              {
                                  {{{128, 1, 8192}, {128, 1, 8192}}, ge::DT_INT32, ge::FORMAT_ND}, // x (invalid dtype)
                                  {{{4, 8192}, {4, 8192}}, ge::DT_INT32, ge::FORMAT_ND},           // weight
                                  {{{128, 3, 8192}, {128, 3, 8192}}, ge::DT_INT32, ge::FORMAT_ND}, // conv_states
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // query_start_loc
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cache_indices
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // initial_state_mode
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // num_accepted_tokens
                              },
                              {
                                  {{{128, 3, 8192}, {128, 3, 8192}}, ge::DT_INT32, ge::FORMAT_ND}, // conv_states
                                  {{{128, 1, 8192}, {128, 1, 8192}}, ge::DT_INT32, ge::FORMAT_ND}, // y
                              },
                              a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dFnTiling, reject_kernel_width_5)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p(
        "CausalConv1d",
        {
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{5, 8192}, {5, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // weight (K=5, invalid)
            {{{2, 4, 8192}, {2, 4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states (stateLen=4 for K=5)
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},               // bias
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8192}, {2, 4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dFnTiling, reject_state_len_too_small)
{
    optiling::CausalConv1d::CausalConv1dCompileInfo c{};
    auto a = MakeAttrs("none", -1);
    gert::TilingContextPara p(
        "CausalConv1d",
        {
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // weight (K=4)
            {{{2, 2, 8192}, {2, 2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states (stateLen=2 < K-1=3)
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},               // bias
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 8192}, {2, 2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        a, &c, "Ascend950PR_9599", 64, 253952);
    ExecuteTestCase(p, ge::GRAPH_FAILED);
}
