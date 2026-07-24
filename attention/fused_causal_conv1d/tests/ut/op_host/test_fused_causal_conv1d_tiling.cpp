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
 * \file test_fused_causal_conv1d_tiling.cpp
 * \brief Unit tests for FusedCausalConv1d tiling logic (BH + BSH, all scenarios)
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/fused_causal_conv1d_cut_bh_tiling_arch35.h"
#include "../../../op_host/fused_causal_conv1d_cut_bsh_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// ============================================================
// Common helpers
// ============================================================

static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(int64_t activationMode = 0, int64_t padSlotId = -1,
                                                              int64_t runMode = 0, int64_t maxQueryLen = -1,
                                                              int64_t residualConnection = 0, int64_t blockSize = 128,
                                                              int64_t convMode = 0)
{
    return {
        {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activationMode)},
        {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(padSlotId)},
        {"run_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(runMode)},
        {"max_query_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxQueryLen)},
        {"residual_connection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(residualConnection)},
        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"conv_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(convMode)},
    };
}

// ============================================================
// BH Tiling Tests (max_query_len <= 8, 3D/2D short-seq)
// ============================================================

class FusedCausalConv1dCutBHTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedCausalConv1dCutBHTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FusedCausalConv1dCutBHTiling TearDown" << std::endl;
    }
};

// ---- 3D Input, BF16, decode baseline ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_3d_bf16_b4_s1_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 1, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // x
                                  {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // weight
                                  {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // queryStartLoc (3D: N/A)
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                // cacheIndices
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                // initialStateMode
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                   // bias
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // initialStateIdx
                              },
                              {
                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // y
                                  {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20000,
                    "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 1 0 512 3 2 512 1024 512 -1 0 1 0 0 "
                    "128 0 1 0 0 0 0 1 128 128 ",
                    {});
}

TEST_F(FusedCausalConv1dCutBHTiling, bh_3d_bf16_b1_s4_d1024)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // x
                                  {{{3, 1024}, {3, 1024}}, ge::DT_BF16, ge::FORMAT_ND},       // weight
                                  {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // queryStartLoc
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // cacheIndices
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // initialStateMode
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                     // bias
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // initialStateIdx
                              },
                              {
                                  {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // y
                                  {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20000,
                    "1 1 1 1 0 1024 1024 1 0 1 1 1 1 1 1 1024 1024 1 1 1 1 1024 1024 1 4 0 1024 3 5 1024 5120 1024 -1 "
                    "0 1 0 0 128 0 1 0 0 0 0 1 1024 1024 ",
                    {});
}

// ---- 2D Input, FP16, non-APC + non-MTP ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_fp16_b1_s4_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{2, 5, 512}, {2, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices 1D
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // numAcceptedTokens (no MTP)
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // initialStateIdx
                              },
                              {
                                  {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // y
                                  {{{2, 5, 512}, {2, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "4 4 1 4 0 128 128 1 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 1 4 4 512 3 5 512 2560 512 -1 1 0 0 0 "
                    "128 0 1 0 0 0 0 1 128 128 ",
                    {});
}

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_fp16_b32_s128_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{128, 512}, {128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // weight
                                  {{{64, 5, 512}, {64, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{33}, {33}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
                                  {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                     // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // initialStateIdx
                              },
                              {
                                  {{{128, 512}, {128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
                                  {{{64, 5, 512}, {64, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "64 4 16 4 0 128 128 16 0 2 2 1 1 2 2 128 128 1 1 2 2 128 128 32 4 128 512 3 5 512 2560 512 -1 1 0 "
                    "0 0 128 0 1 0 0 0 0 1 128 128 ",
                    {});
}

// ---- BH 2D: with MTP ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_hasNAT_fp16_b4_s16_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 0);
    gert::TilingContextPara p(
        "FusedCausalConv1d",
        {
            {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x
            {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
            {{{8, 7, 512}, {8, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates (stateLen=7 for MTP)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // numAcceptedTokens (MTP)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numComputedTokens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxFirst
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxLast
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateIdx
        },
        {
            {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
            {{{8, 7, 512}, {8, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
        },
        a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 4 16 512 3 7 512 3584 512 -1 1 1 0 0 "
                    "128 0 1 0 0 0 0 1 128 128 ",
                    {});
}

// ---- BH 2D: with APC ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_apc_fp16_b8_s32_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{32, 512}, {32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
                                  {{{100, 5, 512}, {100, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND},                       // queryStartLoc
                                  {{{8, 11}, {8, 11}}, ge::DT_INT32, ge::FORMAT_ND}, // cacheIndices 2D (APC)
                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},         // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},           // numAcceptedTokens
                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // numComputedTokens (APC: required)
                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxFirst (APC)
                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxLast  (APC)
                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // initialStateIdx (APC)
                              },
                              {
                                  {{{32, 512}, {32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // y
                                  {{{100, 5, 512}, {100, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "32 4 8 4 0 128 128 8 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 8 4 32 512 3 5 512 2560 512 -1 1 0 0 1 "
                    "128 11 1 0 0 1 0 1 128 128 ",
                    {});
}

// ---- BH 2D: APC + MTP ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_apc_hasNAT_fp16_b6_s48_d768)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 8, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{48, 768}, {48, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
                                  {{{3, 768}, {3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // weight
                                  {{{80, 7, 768}, {80, 7, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},                     // queryStartLoc
                                  {{{6, 11}, {6, 11}}, ge::DT_INT32, ge::FORMAT_ND},             // cacheIndices 2D
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                     // bias
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}, // numAcceptedTokens (MTP)
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}, // numComputedTokens (APC)
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxFirst
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}, // blockIdxLast
                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}, // initialStateIdx
                              },
                              {
                                  {{{80, 7, 768}, {80, 7, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{48, 768}, {48, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // y
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "36 6 6 6 0 128 128 6 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 6 8 48 768 3 7 768 5376 768 -1 1 1 0 1 "
                    "128 11 1 0 0 1 0 1 128 128 ",
                    {});
}


// ---- BH 2D: conv_mode=1 + numComputed ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_conv1_fp16_b4_s16_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 0, 128, 1);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{8, 7, 512}, {8, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numAcceptedTokens
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}, // numComputedTokens (conv_mode=1)
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // initialStateIdx
                              },
                              {
                                  {{{8, 7, 512}, {8, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 4 16 512 3 7 512 3584 512 -1 1 0 0 0 "
                    "128 0 1 0 1 1 0 1 128 128 ",
                    {});
}

// ---- BH 2D: residual_connection ----

TEST_F(FusedCausalConv1dCutBHTiling, bh_2d_residual_fp16_b4_s16_d512)
{
    optiling::FusedCausalConv1dCutBHCompileInfo c = {64, 261888};
    auto a = MakeAttrs(0, -1, 0, 4, 1, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{8, 5, 512}, {8, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateIdx
                              },
                              {
                                  {{{8, 5, 512}, {8, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{16, 512}, {16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
                              },
                              a, &c);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 20001,
                    "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 4 16 512 3 5 512 2560 512 -1 1 0 1 0 "
                    "128 0 1 0 0 0 0 1 128 128 ",
                    {});
}

// ============================================================
// BSH Tiling Tests (max_query_len > 8, 2D input long-seq)
// ============================================================

class FusedCausalConv1dCutBSHTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedCausalConv1dCutBSHTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FusedCausalConv1dCutBSHTiling TearDown" << std::endl;
    }
};

struct BshCompileInfo {};

// ---- BSH: non-APC, non-MTP baseline ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_noapc_fp16_b2_s256_d512)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 256, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // x
                                  {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateIdx
                              },
                              {
                                  {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // y
                              },
                              a, &ci);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "1 1 18 18 128 128 1 1 17 17 128 128 4 0 128 128 16 14 18 17 64 3 256 512 2 -1 2 512 1024 512 0 0 "
                    "128 0 0 0 0 0 1 0 1 128 128 ",
                    {16777216});
}

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_noapc_fp16_b1_s64_d256)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 64, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // x
                                  {{{3, 256}, {3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{2, 2, 256}, {2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                   // queryStartLoc
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                   // cacheIndices
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                   // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateIdx
                              },
                              {
                                  {{{2, 2, 256}, {2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // y
                              },
                              a, &ci);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "1 1 3 3 256 256 1 1 3 3 256 256 1 0 256 256 62 0 3 3 62 3 64 256 1 -1 2 256 512 256 0 0 128 0 0 0 "
                    "0 0 1 0 1 256 256 ",
                    {16777216});
}

// ---- BSH: APC ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_apc_fp16_b12_s200_d1024)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 200, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{200, 1024}, {200, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
                                  {{{3, 1024}, {3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
                                  {{{200, 2, 1024}, {200, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{13}, {13}}, ge::DT_INT32, ge::FORMAT_ND},                       // queryStartLoc
                                  {{{12, 11}, {12, 11}}, ge::DT_INT32, ge::FORMAT_ND}, // cacheIndices 2D (APC)
                                  {{{12}, {12}}, ge::DT_INT32, ge::FORMAT_ND},         // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},             // numAcceptedTokens
                                  {{{12}, {12}}, ge::DT_INT32, ge::FORMAT_ND},         // numComputedTokens
                                  {{{12}, {12}}, ge::DT_INT32, ge::FORMAT_ND},         // blockIdxFirst
                                  {{{12}, {12}}, ge::DT_INT32, ge::FORMAT_ND},         // blockIdxLast
                                  {{{12}, {12}}, ge::DT_INT32, ge::FORMAT_ND},         // initialStateIdx
                              },
                              {
                                  {{{200, 2, 1024}, {200, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{200, 1024}, {200, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // y
                              },
                              a, &ci);
    // TODO: fill expectTilingKey and expectTilingData after first run
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "1 1 24 24 256 256 1 1 24 24 128 128 7 1 256 128 9 0 24 24 63 3 200 1024 12 -1 2 1024 2048 1024 0 "
                    "1 128 11 0 0 0 1 1 0 1 128 128 ",
                    {16777216});
}

// ---- BSH: APC + MTP (key debug case: b58_s243_d2592, conv_mode=1) ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_hasNAT_apc_conv1_fp16_b58_s243_d2592)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 15, 0, 86, 1);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{243, 2592}, {243, 2592}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
                                  {{{3, 2592}, {3, 2592}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
                                  {{{700, 3, 2592}, {700, 3, 2592}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{59}, {59}}, ge::DT_INT32, ge::FORMAT_ND},                       // queryStartLoc
                                  {{{58, 11}, {58, 11}}, ge::DT_INT32, ge::FORMAT_ND}, // cacheIndices 2D (APC)
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // bias
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // numAcceptedTokens (MTP)
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // numComputedTokens
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // blockIdxFirst
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // blockIdxLast
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},         // initialStateIdx
                              },
                              {
                                  {{{700, 3, 2592}, {700, 3, 2592}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{243, 2592}, {243, 2592}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // y
                              },
                              a, &ci);
    // TODO: fill expectTilingKey and expectTilingData after first run
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "1 1 63 63 256 256 1 1 62 62 128 128 16 4 256 128 4 1 63 62 64 3 243 2592 58 -1 3 2592 7776 2592 0 "
                    "1 86 11 1 0 1 1 1 32 2 128 32 ",
                    {16777216});
}

// ---- BSH: conv_mode=1 + numComputed (Plan A debug case: b58_s504_d3296) ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_noapc_conv1_fp16_b58_s504_d3296)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 27, 0, 234, 1);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{504, 3296}, {504, 3296}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
                                  {{{3, 3296}, {3, 3296}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
                                  {{{700, 7, 3296}, {700, 7, 3296}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{59}, {59}}, ge::DT_INT32, ge::FORMAT_ND},                       // queryStartLoc
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},                       // cacheIndices 1D
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND},                       // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                         // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},     // numAcceptedTokens
                                  {{{58}, {58}}, ge::DT_INT32, ge::FORMAT_ND}, // numComputedTokens (conv_mode=1)
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},     // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},     // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},     // initialStateIdx
                              },
                              {
                                  {{{700, 7, 3296}, {700, 7, 3296}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{504, 3296}, {504, 3296}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // y
                              },
                              a, &ci);
    // TODO: fill expectTilingKey and expectTilingData after first run
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "2 1 242 13 128 128 2 1 242 13 128 128 25 0 128 128 2 0 253 253 50 3 504 3296 58 -1 7 3296 23072 "
                    "3296 0 0 234 0 1 0 0 1 1 96 2 128 96 ",
                    {16777216});
}


// ---- BSH: residual_connection ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_noapc_residual_fp16_b4_s128_d1024)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 128, 1, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{128, 1024}, {128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // x
                                  {{{3, 1024}, {3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                  {{{8, 2, 1024}, {8, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},                     // queryStartLoc
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                     // cacheIndices
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                     // initialStateMode
                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                     // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // initialStateIdx
                              },
                              {
                                  {{{8, 2, 1024}, {8, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // convStates
                                  {{{128, 1024}, {128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // y
                              },
                              a, &ci);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10001,
                    "1 1 16 16 256 256 1 1 16 16 128 128 7 1 256 128 9 0 16 16 63 3 128 1024 4 -1 2 1024 2048 1024 1 0 "
                    "128 0 0 0 0 0 1 0 1 128 128 ",
                    {16777216});
}

// ---- BSH: BF16 ----

TEST_F(FusedCausalConv1dCutBSHTiling, bsh_noNAT_noapc_bf16_b4_s128_d512)
{
    BshCompileInfo ci;
    auto a = MakeAttrs(0, -1, 0, 128, 0, 128, 0);
    gert::TilingContextPara p("FusedCausalConv1d",
                              {
                                  {{{128, 512}, {128, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // x
                                  {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // weight
                                  {{{8, 2, 512}, {8, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},                // queryStartLoc
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                // cacheIndices
                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},                // initialStateMode
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                   // bias
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // numAcceptedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // numComputedTokens
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // blockIdxFirst
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // blockIdxLast
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // initialStateIdx
                              },
                              {
                                  {{{8, 2, 512}, {8, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // convStates
                                  {{{128, 512}, {128, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // y
                              },
                              a, &ci);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 10000,
                    "1 1 8 8 256 256 1 1 8 8 128 128 3 1 256 128 21 0 8 8 63 3 128 512 4 -1 2 512 1024 512 0 0 128 0 0 "
                    "0 0 0 1 0 1 128 128 ",
                    {16777216});
}
