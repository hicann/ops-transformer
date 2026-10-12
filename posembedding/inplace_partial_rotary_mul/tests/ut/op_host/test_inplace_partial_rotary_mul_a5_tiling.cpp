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
 * @file test_inplace_partial_rotary_mul_tiling_a5.cpp
 * @brief OpHost UT for InplacePartialRotaryMul on Ascend950 (RegBase tiling)
 *
 * Covers 5 TilingClass strategies in priority order:
 *   AAndB (2004x) > AB (2003x) > ABAAndBA (2001x) > BAB (2002x)
 *
 * TilingKey encoding:
 *   AAndB: 2004x, x=0(A/NO_BROADCAST), x=1(B/BROADCAST_BSN)
 *   BAB:   20020
 *   AB:    20030
 *   ABA:   2001x, x=0(ABA/cosB!=1), x=1(BA/cosB==1)
 *   Mixed precision offset: BF16+FP32=+100, FP16+FP32=+200
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/inplace_partial_rotary_mul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class InplacePartialRotaryMulTilingA5 : public testing::Test {
protected:
    // Ascend950 hardware parameters
    static constexpr uint64_t k950CoreNum = 64;
    static constexpr uint64_t k950UbSize = 253952; // 248KB
    static constexpr const char* kSocVersion = "Ascend950";

    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulTilingA5 SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulTilingA5 TearDown" << endl;
    }
};

// ============================================================================
// 1. AAndB TilingClass — NO_BROADCAST layout (TilingKey A: 20040)
// ============================================================================

// AAndB-01: FP16, NO_BROADCAST, full D
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_no_broadcast_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-02: BF16, NO_BROADCAST, full D
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_no_broadcast_bf16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-03: FP32, NO_BROADCAST, full D
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_no_broadcast_fp32)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-04: FP16+FP32 Mixed Precision, NO_BROADCAST → TilingKey A_FP16_FP32 (20240)
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_no_broadcast_fp16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20240;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-05: BF16+FP32 Mixed Precision, NO_BROADCAST → TilingKey A_BF16_FP32 (20140)
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_no_broadcast_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20140;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 2. AAndB TilingClass — BROADCAST_BSN layout (TilingKey B: 20041)
//    cos shape = [1,1,1,D], 完全广播到 x=[B,S,N,D]
// ============================================================================

// AAndB-06: FP16, BROADCAST_BSN
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_broadcast_bsn_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20041;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-07: BF16+FP32 Mixed, BROADCAST_BSN → TilingKey B_BF16_FP32 (20141)
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_broadcast_bsn_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20141;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AAndB-08: FP16+FP32 Mixed, BROADCAST_BSN → TilingKey B_FP16_FP32 (20241)
TEST_F(InplacePartialRotaryMulTilingA5, AAndB_broadcast_bsn_fp16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20241;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 3. AAndB — 111D layout (all dims=1 except D) → still NO_BROADCAST via JudgeLayoutByShape
//    x=[1,1,1,64], cos=[1,1,1,64] → shape all equal → NO_BROADCAST → Key A
// ============================================================================

TEST_F(InplacePartialRotaryMulTilingA5, AAndB_111d_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 4. AAndB — partial_slice 非全量
//    x=[2,4,8,128], partial_slice=[32,96] → sliceLength=64
//    cos/sin last dim must = 64
// ============================================================================

TEST_F(InplacePartialRotaryMulTilingA5, AAndB_partial_slice)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 5. BAB TilingClass — BSND layout (cos=[1,S,1,D])
//    Only works when cosB == 1
// ============================================================================

TEST_F(InplacePartialRotaryMulTilingA5, BAB_bsnd_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20020;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// BAB: BF16+FP32 Mixed → TilingKey BAB_BF16_FP32 (20120)
TEST_F(InplacePartialRotaryMulTilingA5, BAB_bsnd_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20120;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// BAB: FP16+FP32 Mixed → TilingKey BAB_FP16_FP32 (20220)
TEST_F(InplacePartialRotaryMulTilingA5, BAB_bsnd_fp16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20220;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 6. BAB — BAB IsCapable 不满足时 falls through to AAndB (cosB==B, not 1)
//    x=[2,4,8,64], cos=[2,4,1,64] → JudgeLayoutByShape: BSND, cosB==B==2, not 1
//    BAB IsCapable returns false (cosB != 1), so falls to next priority
//    AAndB IsCapable returns false (layout is BSND not NO_BROADCAST/BROADCAST_BSN)
//    AB IsCapable returns false (not SBND)
//    ABAAndBA IsCapable returns false (not BNSD)
//    → all return false, fallback to 910B path!
// ============================================================================

// BAB: 大 S 场景
TEST_F(InplacePartialRotaryMulTilingA5, BAB_large_s)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 32, 16, 64}, {1, 32, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 32, 16, 64}, {1, 32, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20020;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 7. AB TilingClass — SBND layout (x=[S,B,N,D], cos=[S,B,1,D] or [S,1,1,D])
//    AB IsCapable: layout == SBND
// ============================================================================

// AB: SBND with S11D broadcast (cos=[S,1,1,D])
TEST_F(InplacePartialRotaryMulTilingA5, AB_sbnd_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20030;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AB: SBND BF16+FP32 Mixed
TEST_F(InplacePartialRotaryMulTilingA5, AB_sbnd_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20130;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AB: SBND FP16+FP32 Mixed
TEST_F(InplacePartialRotaryMulTilingA5, AB_sbnd_fp16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 2, 8, 64}, {4, 2, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20230;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 8. ABAAndBA TilingClass — BNSD layout (x=[B,N,S,D], cos=[1,N,1,D] or [B,N,1,D])
//    ABA (cosB!=1), BA (cosB==1)
// ============================================================================

// ABA shape: x=[2,8,4,64] cos=[2,8,1,64] → SBND matched by AB first (priority > ABA) → Key AB (20030)
TEST_F(InplacePartialRotaryMulTilingA5, ABA_bnsd_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 1, 64}, {2, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 1, 64}, {2, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20030;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// BA shape: x=[2,8,4,64] cos=[1,8,1,64] → BSND with cosB=1 → BAB (20020)
// BAB has higher priority than ABAAndBA
TEST_F(InplacePartialRotaryMulTilingA5, BA_bnsd_cosB1_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20020;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// BAB BF16+FP32 Mixed → Key BAB_BF16_FP32 (20120)
TEST_F(InplacePartialRotaryMulTilingA5, BA_bnsd_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20120;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// AB BF16+FP32 Mixed → x=[2,8,4,64] cos=[2,8,1,64] → SBND → AB → Key AB_BF16_FP32 (20130)
TEST_F(InplacePartialRotaryMulTilingA5, ABA_bnsd_bf16_cos_fp32_mixed)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 1, 64}, {2, 8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 1, 64}, {2, 8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4, 64}, {2, 8, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20130;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 9. 边界值测试
// ============================================================================

// D=2 最小值
TEST_F(InplacePartialRotaryMulTilingA5, boundary_d_min)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 1, 2}, {1, 1, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 2}, {1, 1, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 2}, {1, 1, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 2}, {1, 1, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 2})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// D=1024 最大值
TEST_F(InplacePartialRotaryMulTilingA5, boundary_d_max)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1024}, {1, 1, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 1024})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// 大 B (多核切分验证)
TEST_F(InplacePartialRotaryMulTilingA5, boundary_large_b)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{32, 1, 16, 64}, {32, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 16, 64}, {32, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 16, 64}, {32, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 1, 16, 64}, {32, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 10. 参数校验 — 非法输入（预期 GRAPH_FAILED）
// ============================================================================

// mode != 1 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_mode)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// D > 1024 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_d_too_large)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 1, 2048}, {1, 1, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 2048}, {1, 1, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 2048}, {1, 1, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 2048}, {1, 1, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 2048})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// D 为奇数 (interleave requires D%2==0) 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_d_odd)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 1, 63}, {1, 1, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 63}, {1, 1, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 63}, {1, 1, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 63}, {1, 1, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 63})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// sliceStart < 0 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_slice_start_negative)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({-1, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// Empty slice (sliceStart == sliceEnd): valid no-op, no rope computation
TEST_F(InplacePartialRotaryMulTilingA5, slice_empty_noop)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 32})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    // Empty slice (start==end) is valid: kernel does no rope computation
    uint64_t expectTilingKey = 20040; // TILING_KEY_A: regbase no-op key (kernel checks sliceLength==0)
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// cos/sin last dim == 0: valid no-op, no rope computation
TEST_F(InplacePartialRotaryMulTilingA5, cos_sin_empty_d_noop)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 0}, {2, 4, 8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 0}, {2, 4, 8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 20040;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// cos/sin last dim != sliceLength 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_cos_lastdim_mismatch_slice)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    // cos last dim = 128 != sliceLength = 64
    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// sliceLength 为奇数 (interleave requires sliceLength%2==0) 非法
// x D=128 (even, passes D%2 check), but slice=[0,3) → sliceLength=3 (odd)
TEST_F(InplacePartialRotaryMulTilingA5, invalid_slice_length_odd)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 3}, {2, 4, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 3}, {2, 4, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 3})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// 不支持的 dtype 组合 (x=FP32, cos=BF16)
TEST_F(InplacePartialRotaryMulTilingA5, invalid_dtype_combination)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// cos shape != sin shape 非法
TEST_F(InplacePartialRotaryMulTilingA5, invalid_cos_sin_shape_mismatch)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k950CoreNum, k950UbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}
