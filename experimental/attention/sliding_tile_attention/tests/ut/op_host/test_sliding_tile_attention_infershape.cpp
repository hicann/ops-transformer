/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <cmath>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class sliding_tile_attention_infershape_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sliding_tile_attention_infershape_ut SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sliding_tile_attention_infershape_ut TearDown" << endl;
    }
};

namespace {
// Canvas 36x48x48: 6x6x6 tiles of 6x8x8 tokens, so the image holds 82944 tokens and no text follows.
constexpr int64_t batch = 1;
constexpr int64_t numHeads = 8;
constexpr int64_t headDim = 64;
constexpr int64_t imageTokens = 36 * 48 * 48;
constexpr int64_t seqLen = imageTokens;
constexpr int64_t windowDims = 3;
constexpr int64_t tileGrid = 6 | (6LL << 16) | (6LL << 32);
constexpr int64_t maxSelectNum = 27;
constexpr float scaleValue = 1.0F / sqrt(static_cast<float>(headDim));

// Canvas 30x48x80 with 128 text tokens: 5x6x10 tiles, 115200 image tokens.
constexpr int64_t textImageTokens = 30 * 48 * 80;
constexpr int64_t textLength = 128;
constexpr int64_t textSeqLen = textImageTokens + textLength;
constexpr int64_t textTileGrid = 5 | (6LL << 16) | (10LL << 32);
constexpr int64_t textMaxSelectNum = 27 + 1;
} // namespace

// ============================================================================
// 用例 1: BNSD 无文本正常场景，输出形状与 query 一致
// ============================================================================
TEST_F(sliding_tile_attention_infershape_ut, bnsd_no_text_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SlidingTileAttention",
        {{{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, numHeads, 1, windowDims}, {1, numHeads, 1, windowDims}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"validKvSeqLen", Ops::Transformer::AnyValue::CreateFrom<int64_t>(seqLen)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"originalHeadDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headDim)},
         {"tileGrid", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tileGrid)},
         {"maxSelectNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSelectNum)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, seqLen, headDim}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 2: BNSD 含文本区正常场景，文本预留槽计入 S，输出形状仍与 query 一致
// ============================================================================
TEST_F(sliding_tile_attention_infershape_ut, bnsd_with_text_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SlidingTileAttention",
        {{{{batch, numHeads, textSeqLen, headDim}, {batch, numHeads, textSeqLen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, textSeqLen, headDim}, {batch, numHeads, textSeqLen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, textSeqLen, headDim}, {batch, numHeads, textSeqLen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{1, numHeads, 1, windowDims}, {1, numHeads, 1, windowDims}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"validKvSeqLen", Ops::Transformer::AnyValue::CreateFrom<int64_t>(textImageTokens + textLength)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"originalHeadDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headDim)},
         {"tileGrid", Ops::Transformer::AnyValue::CreateFrom<int64_t>(textTileGrid)},
         {"maxSelectNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(textMaxSelectNum)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, textSeqLen, headDim}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 3: BFLOAT16 正常场景
// ============================================================================
TEST_F(sliding_tile_attention_infershape_ut, bnsd_bf16_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SlidingTileAttention",
        {{{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, seqLen, headDim}, {batch, numHeads, seqLen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, numHeads, 1, windowDims}, {1, numHeads, 1, windowDims}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"validKvSeqLen", Ops::Transformer::AnyValue::CreateFrom<int64_t>(seqLen)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"originalHeadDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headDim)},
         {"tileGrid", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tileGrid)},
         {"maxSelectNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSelectNum)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, seqLen, headDim}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 4: query 不是四维 BNSD，应推导失败
// ============================================================================
TEST_F(sliding_tile_attention_infershape_ut, query_not_4d_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SlidingTileAttention",
        {{{{numHeads, seqLen, headDim}, {numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{numHeads, seqLen, headDim}, {numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{numHeads, seqLen, headDim}, {numHeads, seqLen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, numHeads, 1, windowDims}, {1, numHeads, 1, windowDims}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"validKvSeqLen", Ops::Transformer::AnyValue::CreateFrom<int64_t>(seqLen)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"originalHeadDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headDim)},
         {"tileGrid", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tileGrid)},
         {"maxSelectNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSelectNum)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{numHeads, seqLen, headDim}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
