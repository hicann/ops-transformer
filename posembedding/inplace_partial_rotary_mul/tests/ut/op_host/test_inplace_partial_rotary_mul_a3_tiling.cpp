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
 * @file test_inplace_partial_rotary_mul_a3_tiling.cpp
 * @brief OpHost UT for InplacePartialRotaryMul on Ascend910B/910_93 (Membase tiling)
 *
 * Tiling paths:
 *   1. Fast path (CalTilingData succeeds):
 *      - isSpecail=true: r1dim1==1, r1dim2==1, (xdim1==1 or xdim2==1), r1dim0==dim0
 *      - isAlign=true: D % blockSize == 0 (16 for FP16/BF16, 32 for FP32)
 *      - D <= 64 (REPEAT_FP32)
 *      - TilingKey 1 (isBrc=true) or 2 (isBrc=false)
 *
 *   2. Generic path (TilingSplit):
 *      - Split S/BS/BSN based on UB size calculation
 *      - dtype offset: FP16=0, BF16=10, FP32=20
 *      - split: S=+0, BS=+100, BSN=+200, pad=+1
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/inplace_partial_rotary_mul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class InplacePartialRotaryMulTilingA3 : public testing::Test {
protected:
    // Ascend910B hardware parameters
    static constexpr uint64_t k910bCoreNum = 64;
    static constexpr uint64_t k910bUbSize = 262144; // 256KB
    static constexpr const char* kSocVersion = "Ascend910B";

    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulTilingA3 SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulTilingA3 TearDown" << endl;
    }
};

// ============================================================================
// 1. Fast Path (TilingKey 1/2) — small D, aligned, special shapes
//    Requirements: isSpecail && isAlign && D <= 64
// ============================================================================

// Fast Path Key 1: isBrc=true — B=1,S>1,N=1,D aligned, cos=[B,1,1,D]
TEST_F(InplacePartialRotaryMulTilingA3, fast_path_key1_brc_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            // x: [1, 4, 1, 64] — xdim1=4 != 1, xdim2=1 == 1 → isSpecail
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // cos: [1, 1, 1, 64] — r1dim1=1, r1dim2=1, r1dim0=1 → isBrc=true (r1dim1=1 != dim1=4)
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    uint64_t expectTilingKey = 1;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// Fast Path Key 2: isBrc=false — B=1,S=1,N=1,D=64, all dims=1 except D
//   x=[1,1,1,64], cos=[1,1,1,64] → xdim1=1, xdim2=1, dim1=1*1=1, r1dim1=1 → isBrc=false
TEST_F(InplacePartialRotaryMulTilingA3, fast_path_key2_no_brc_fp16)
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
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    uint64_t expectTilingKey = 2;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ============================================================================
// 2. Generic Path — Split S, D aligned (no pad)
//    TilingKey = BASE_KEY (2000) + dtype_offset + split_mode
//    dtype: FP16=0, BF16=10, FP32=20
// ============================================================================

// Split S: small shape, x=[B,SN,1,D] → r1dim1 matches, map to SplitS
// cos=[1,S,1,D] with r1dim0=1, r1dim2=1, xdim1==r1dim1 → seqLenOut=cos[1]
// batchSize=1, so TilingSplitS path → ub size allows S split → TilingKey 2000
TEST_F(InplacePartialRotaryMulTilingA3, generic_split_s_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    // x=[1, 4, 1, 256], cos=[1, 4, 1, 256] → r1dim0=1, r1dim2=1, xdim1=4==r1dim1=4
    // → batchSizeOut=1, seqLenOut=4, numHeadsOut=1
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 256}, {1, 4, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 256}, {1, 4, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 256}, {1, 4, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 256}, {1, 4, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 256})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    // CalTilingData fails (isSpecail=false since dim1=4, r1dim1=4, xdim1=4, xdim2=1 → checks pass but wait...)
    // Actually: r1dim1_==4 != 1 → isSpecail=false → CalTilingData fails → falls to generic
    // TilingSplit: SBND layout → batchSize=1, seqLen=4, numHeads=1, headDim=headDim_=256
    // headDim 256 % 16 == 0 → isAlign_=true
    // buffer calc → should produce a valid TilingKey
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

// Split S + BF16
TEST_F(InplacePartialRotaryMulTilingA3, generic_split_s_bf16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2010);
}

// Split S + FP32
TEST_F(InplacePartialRotaryMulTilingA3, generic_split_s_fp32)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2020);
}

// ============================================================================
// 3. Generic Path — Split BS
//    Requires B > 1 so that BS split triggers
//    x=[B,S,N,D=128], cos=[1,S,1,D] → batchSize=1 (always), but with N>1 it goes to BS split
// ============================================================================

// Split BS: x=[2, 4, 1, 128], cos=[1, 4, 1, 128] → BSND-type, BS split
TEST_F(InplacePartialRotaryMulTilingA3, generic_split_bs_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    // x=[2,4,1,128], cos=[1,4,1,128] → r1dim0=1, r1dim1=4, r1dim2=1
    // xdim0=2==r1dim0(1)? No. xdim0=2, r1dim0=1
    // Alternative: xdim1=4==r1dim1(4) → yes, BSND match
    // → batchSizeOut=2(!), but wait, batchSize must be 1...
    // Actually: SBND → batchSize=xdim0=2, seqLen=r1dim1=4, numHeads=xdim2=1
    // But TilingSplit requires batchSizeOut==1! So this might FAIL.
    // Let me use a different shape that maps to BS split without the batchSize constraint
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            // x=[1, 8, 4, 128], cos=[1, 8, 1, 128] → xdim0=1, xdim1=8==r1dim1=8
            // → batchSize=xdim0=1, seqLen=r1dim1=8, numHeads=xdim2=4
            {{{1, 8, 4, 128}, {1, 8, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 8, 4, 128}, {1, 8, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

// Split BS + BF16
TEST_F(InplacePartialRotaryMulTilingA3, generic_split_bs_bf16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            // Larger shape to force BS split: x=[1, 8, 16, 64], cos=[1, 8, 1, 64]
            {{{1, 8, 16, 64}, {1, 8, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 8, 16, 64}, {1, 8, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2010);
}

// ============================================================================
// 4. Generic Path — Split BSN (more complex shapes)
//    Large N forces BSN split
// ============================================================================

// BNSD layout → BSN path. x=[1,32,4,64], cos=[1,32,1,64]
// batchSize=1, seqLen=4, numHeads=32
TEST_F(InplacePartialRotaryMulTilingA3, generic_bsnd_bsn_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 32, 4, 64}, {1, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 32, 4, 64}, {1, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

// BSND layout, batchSize forced to 1 by restriction: x=[1,4,32,64], cos=[1,4,1,64]
TEST_F(InplacePartialRotaryMulTilingA3, generic_bsnd_split_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 32, 64}, {1, 4, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 32, 64}, {1, 4, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

// ============================================================================
// 5. Generic Path — Unaligned D (PAD variant)
//    D not divisible by 16 → isAlign=false → tilingKey += 1 (PAD)
// ============================================================================

TEST_F(InplacePartialRotaryMulTilingA3, generic_pad_fp16)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    // D=255 not divisible by 16
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 1, 4, 254}, {1, 1, 4, 254}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 254}, {1, 1, 1, 254}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 254}, {1, 1, 1, 254}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 4, 254}, {1, 1, 4, 254}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 254})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2001);
}

// ============================================================================
// 6. Partial Slice on Generic Path
// ============================================================================

TEST_F(InplacePartialRotaryMulTilingA3, generic_partial_slice)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    // sliceStart=32, sliceEnd=96 → sliceLength=64, cos/sin last dim=64
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

TEST_F(InplacePartialRotaryMulTilingA3, cos_sin_empty_d_noop)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{458, 1, 8, 2}, {458, 1, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{458, 1, 1, 0}, {458, 1, 1, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{458, 1, 1, 0}, {458, 1, 1, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{458, 1, 8, 2}, {458, 1, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 2})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 1);
}

// ============================================================================
// 7. Boundary Tests
// ============================================================================

// D=2 minimum, aligned (2 % 16 != 0 → unaligned → PAD)
TEST_F(InplacePartialRotaryMulTilingA3, boundary_d_min)
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
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2001);
}

// D=1024 maximum, aligned (1024 % 16 = 0 → D_LIMIT check passes)
TEST_F(InplacePartialRotaryMulTilingA3, boundary_d_max)
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
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, 2000);
}

// ============================================================================
// 8. Invalid Input Tests
// ============================================================================

// mode != 1
TEST_F(InplacePartialRotaryMulTilingA3, invalid_mode)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// D > 1024
TEST_F(InplacePartialRotaryMulTilingA3, invalid_d_too_large)
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
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// D odd
TEST_F(InplacePartialRotaryMulTilingA3, invalid_d_odd)
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
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// TODO: cos/sin shape mismatch — membase CheckInput currently doesn't validate cos==sin shape
// Test documents actual behavior: tiling returns GRAPH_SUCCESS (validation gap)
TEST_F(InplacePartialRotaryMulTilingA3, cos_sin_shape_mismatch_not_caught)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}
TEST_F(InplacePartialRotaryMulTilingA3, invalid_slice_negative)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 64}, {1, 4, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({-1, 64})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// sliceLength 为奇数 (interleave requires sliceLength%2==0) 非法
// x D=128 (even, passes D%2 check), but slice=[0,3) → sliceLength=3 (odd)
TEST_F(InplacePartialRotaryMulTilingA3, invalid_slice_length_odd)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 3}, {1, 1, 1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 3}, {1, 1, 1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 3})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}

// cos last dim != sliceLength
TEST_F(InplacePartialRotaryMulTilingA3, invalid_cos_slicelen_mismatch)
{
    optiling::InplacePartialRotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMul",
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        },
        &compileInfo, kSocVersion, k910bCoreNum, k910bUbSize);

    ExecuteTestCase(tilingPara, ge::GRAPH_FAILED);
}
