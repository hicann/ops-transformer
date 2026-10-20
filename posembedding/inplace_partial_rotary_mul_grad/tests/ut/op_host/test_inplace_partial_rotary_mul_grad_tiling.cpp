/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_inplace_partial_rotary_mul_grad_tiling.cpp
 * @brief OpHost UT for InplacePartialRotaryMulGrad tiling (Ascend950, RegBase)
 *
 * Coverage per template (in priority order):
 *   ABA&BA (10000): BNSD  → TilingKey ABA=201 / BA=202
 *   BAB    (20000): BSND  → TilingKey BAB=203
 *   AB     (25000): SBND  → TilingKey AB=204
 *   A&B    (40000): NO_BROADCAST / BROADCAST_BSN → TilingKey A=205 / B=206
 *
 * Dtype combinations: fp16+fp16, fp32+fp32, fp16+fp32, bf16+bf16
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/inplace_partial_rotary_mul_grad_regbase_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class InplacePartialRotaryMulGradTilingTest : public testing::Test {
protected:
    static constexpr uint64_t k950CoreNum = 64;
    static constexpr uint64_t k950UbSize = 253952; // 248KB
    static constexpr const char* kSocVersion = "Ascend950";
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo_ = {};

    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulGradTilingTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulGradTilingTest TearDown" << endl;
    }
};

// ==================== BAB (203): BSND, cosb_==1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy out
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_fp16_fp32_mixed)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 1, 0}, {1, 2, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 1, 0}, {1, 2, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== AB (204): SBND ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, AB_SBND_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [S,B,N,D]
            {{{2, 1, 1, 4}, {2, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [S,1,1,cosD]
            {{{2, 1, 1, 4}, {2, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 1, 0}, {2, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 1, 0}, {2, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 204;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== ABA (201): BNSD, cosb_!=1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, ABA_BNSD_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{2, 1, 2, 4}, {2, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [B,1,S,cosD]
            {{{2, 1, 2, 4}, {2, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 2, 0}, {2, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 2, 0}, {2, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 201;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BA (202): BNSD, cosb_==1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BA_BNSD_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{1, 1, 2, 4}, {1, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,S,cosD]
            {{{1, 1, 2, 4}, {1, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 2, 0}, {1, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 2, 0}, {1, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 202;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== A (205): NO_BROADCAST ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, A_NO_BROADCAST_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos same shape
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 0}, {2, 2, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 0}, {2, 2, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 205;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== B (206): BROADCAST_BSN ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, B_BROADCAST_BSN_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,1,D]
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 0}, {1, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 0}, {1, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 206;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BAB (203) SplitCore: S-priority branch ====================
// usedCoreNum1 * ubFactorS1 < usedCoreNum2 * ubFactorS2 requires ubFactorS_ == 1
// (forced by a large N) so that the comparison degrades to usedCoreNum1 < usedCoreNum2.
TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_split_core_s_priority)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{5, 63, 2000, 8}, {5, 63, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,S,N,D]
            {{{1, 63, 1, 4}, {1, 63, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},       // cos [1,S,1,cosD]
            {{{1, 63, 1, 4}, {1, 63, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},       // sin
        },
        {
            {{{5, 63, 2000, 8}, {5, 63, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== AB (204): SBND, cosb_ != 1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, AB_SBND_cosb_not_one)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [S,B,N,D]
            {{{2, 2, 1, 4}, {2, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [S,B,1,cosD]
            {{{2, 2, 1, 4}, {2, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 204;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== AB (204): SBND, bs > coreNum/2 -> N not split ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, AB_SBND_bs_greater_than_half_core)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{64, 2, 2, 8}, {64, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [S,B,N,D], S=64
            {{{64, 1, 1, 4}, {64, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [S,1,1,cosD]
            {{{64, 1, 1, 4}, {64, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{64, 2, 2, 8}, {64, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 204;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== ABA (201): b % coreNum == 0 -> split on B only ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, ABA_BNSD_b_divisible_by_core)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{64, 2, 2, 8}, {64, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D], B=64
            {{{64, 1, 2, 4}, {64, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [B,1,S,cosD]
            {{{64, 1, 2, 4}, {64, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{64, 2, 2, 8}, {64, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 201;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BA (202): s % coreNum == 0 -> split on S only ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BA_BNSD_s_divisible_by_core)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D], S=64
            {{{1, 1, 64, 4}, {1, 1, 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,S,cosD]
            {{{1, 1, 64, 4}, {1, 1, 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 202;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== ABA (201) SplitCore: S-priority branch ====================
// b=5, s=63: B-priority uses 55 cores, S-priority uses 63 cores -> S-priority wins.
TEST_F(InplacePartialRotaryMulGradTilingTest, ABA_BNSD_split_core_s_priority)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{5, 2, 63, 8}, {5, 2, 63, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{5, 1, 63, 4}, {5, 1, 63, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [B,1,S,cosD]
            {{{5, 1, 63, 4}, {5, 1, 63, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{5, 2, 63, 8}, {5, 2, 63, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 201;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BA (202): 1SND pattern (is1snd_ reshape to 11(S*N)) ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BA_BNSD_1SND_fp16)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{1, 2, 2, 4}, {1, 2, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // cos [1,N,S,cosD]
            {{{1, 2, 2, 4}, {1, 2, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 202;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== A (205): NO_BROADCAST bf16 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, A_NO_BROADCAST_bf16)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 205;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== B (206): BROADCAST_BSN bf16 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, B_BROADCAST_BSN_bf16)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 206;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BAB (203): bf16 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_bf16)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== 403 no-op: empty cos/sin tensor (cosSize == 0, non-empty slice) ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, empty_cos_tensor_noop)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 1, 1, 4}, {0, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 1, 1, 4}, {0, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 403; // TILING_KEY_EMPTY
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== 403 no-op: valid non-zero empty slice [4, 4) ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, nonzero_empty_partial_slice_noop)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 403; // TILING_KEY_EMPTY
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== CheckAttr: unsupported rotary_mode values ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_mode_half)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_mode_quarter)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_mode_interleave_half)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_mode_out_of_range)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckAttr: missing attrs ====================
// No rotary_mode attr -> mode pointer is nullptr -> defaults to 0 -> rejected
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_missing_rotary_mode_attr)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// Only rotary_mode given -> GetListInt(1) returns nullptr -> rejected
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_missing_partial_slice_attr)
{
    gert::TilingContextPara tilingPara("InplacePartialRotaryMulGrad",
                                       {
                                           {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                       },
                                       &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckAttr: slice range / relation failures ====================
// start > end
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_partial_slice_start_greater_than_end)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({6, 2})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// slice length != cos D
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_slice_length_not_equal_cos_d)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// odd slice length under interleave mode
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_odd_slice_length_interleave)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 3}, {1, 1, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 3}, {1, 1, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 3})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// dy D axis over the 1024 limit
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_dy_d_over_limit)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 1, 1, 1032}, {1, 1, 1, 1032}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1032}, {1, 1, 1, 1032}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 1024})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckShapeDim: non-4D tensors ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_3d_dy)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 8}, {2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 8}, {2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_3d_cos)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4}, {1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_3d_sin)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4}, {1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_3d_output)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 8}, {2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckShapeLimit: cos != sin ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_cos_sin_shape_mismatch)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckShapeLimit: input dy != output dy ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_dy_out_shape_mismatch)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckShape: cos D > dy D (only reachable on the no-op path,
// where the sliceLength == cosD check is skipped) ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_cos_d_greater_than_dy_d_noop)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{0, 2, 2, 4}, {0, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 2, 2, 4}, {0, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 0})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckShapeAllPositive: negative dim ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_negative_dim_dy)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{-1, 2, 2, 8}, {-1, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{-1, 2, 2, 8}, {-1, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== CheckDtype failures ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_unsupported_dy_dtype)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_dy_out_dtype_mismatch)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_unsupported_cos_dtype)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_sin_cos_dtype_mismatch)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// ==================== JudgeLayoutByShape: no valid broadcast pattern ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_invalid_cos_broadcast_layout)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 1, 4}, {3, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 1, 4}, {3, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// Issue #4531: out-of-range empty slice must be rejected even though it hits the no-op path
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_out_of_range_empty_partial_slice)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({9, 9})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// Issue #4531: negative start must be rejected even when a tensor is empty (no-op path)
TEST_F(InplacePartialRotaryMulGradTilingTest, rejects_negative_start_partial_slice_with_empty_tensor)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{0, 2, 2, 8}, {0, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 2, 2, 8}, {0, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({-1, 0})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// Issue #4531: legal empty slice [0, 0) keeps the no-op behavior
TEST_F(InplacePartialRotaryMulGradTilingTest, legal_empty_partial_slice_noop)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 0})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 403; // TILING_KEY_EMPTY
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// Issue #4531: legal empty tensor keeps the no-op behavior
TEST_F(InplacePartialRotaryMulGradTilingTest, legal_empty_tensor_noop)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{0, 2, 2, 8}, {0, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 2, 2, 8}, {0, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 0})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 403; // TILING_KEY_EMPTY
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}
