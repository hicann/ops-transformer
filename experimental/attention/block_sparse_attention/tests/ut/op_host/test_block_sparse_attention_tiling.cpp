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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class block_sparse_attention_tiling_ut : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "block_sparse_attention_tiling_ut SetUp" << endl; }
    static void TearDownTestCase() { cout << "block_sparse_attention_tiling_ut TearDown" << endl; }
};

namespace {
constexpr int64_t batch = 1;
constexpr int64_t qSeqlen = 128;
constexpr int64_t kvSeqlen = 128;
constexpr int64_t numHeads = 2;
constexpr int64_t numKvHeads = 1;
constexpr int64_t headDim = 128;
constexpr int64_t blockShapeX = 128;
constexpr int64_t blockShapeY = 128;
int64_t blockShapeData[] = {blockShapeX, blockShapeY};
constexpr int64_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;
constexpr int64_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;
constexpr int64_t tokenQ = batch * qSeqlen;
constexpr int64_t tokenKv = batch * kvSeqlen;
int64_t actualSeqLengthsData[] = {qSeqlen};
int64_t actualSeqLengthsKvData[] = {kvSeqlen};
constexpr float scaleValue = 1.0 / sqrt(static_cast<float>(headDim));
} // namespace

// ============================================================================
// 用例 1: TND正常场景 dtype=fp16 innerPrecise=1
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9000000030100002);
}

// ============================================================================
// 用例 2: BNSD正常场景 dtype=bf16 innerPrecise=0
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bnsd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9000000050022223);
}

// ============================================================================
// 用例 3: BSND正常场景 dtype=fp16 innerPrecise=0 softmaxLseFlag=1
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bsnd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000160400004);
}

// ============================================================================
// 用例 4: TND actualSeqLengths总和与query的T维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_actualSeqLengthsSum_qT_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ + 1, numHeads, headDim}, {tokenQ + 1, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ + 1, numHeads, headDim}, {tokenQ + 1, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ + 1, numHeads, 1}, {tokenQ + 1, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 5: TND actualSeqLengthsKv总和与key/value的T维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_actualSeqLengthsSum_kvT_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv + 1, numKvHeads, headDim}, {tokenKv + 1, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv + 1, numKvHeads, headDim}, {tokenKv + 1, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 6: 不支持的qInputLayout
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, unsupported_qInputLayout)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 7: qInputLayout和kvInputLayout不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, qInputLayout_kvInputLayout_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 8: 非A5使用了BSND格式
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_A5_uses_BSND)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 9: A5不支持的query数据类型
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_unsupported_dtype)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 10: A2A3不支持的query数据类型
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, unsupported_dtype_A2A3)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}},
          ge::DT_FLOAT8_E4M3FN,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT8_E4M3FN,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT8_E4M3FN,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}},
          ge::DT_FLOAT8_E4M3FN,
          ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 11: query与key/value的数据类型不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, qDtype_kvDtype_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 12: query的headDim和key/value的headDim不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, q_headDim_kv_headDim_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim + 1}, {tokenKv, numKvHeads, headDim + 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim + 1}, {tokenKv, numKvHeads, headDim + 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 13: key/value的头数与属性numKeyValueHeads大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, kv_numKvHeads_numKeyValueHeads_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads + 1)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 14: numHeads大小与numKvHeads大小不满足GQA约束
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, gqa_constraint_unsatisfied)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, 4, headDim}, {tokenQ, 4, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, 3, headDim}, {tokenKv, 3, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, 3, headDim}, {tokenKv, 3, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, 4, qBlockNum, kvBlockNum}, {batch, 4, qBlockNum, kvBlockNum}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, 4, headDim}, {tokenQ, 4, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, 4, 1}, {tokenQ, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 15: 不支持的headDim大小
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, unsupported_headDim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, 256}, {tokenQ, numHeads, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, 256}, {tokenKv, numKvHeads, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, 256}, {tokenKv, numKvHeads, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, 256}, {tokenQ, numHeads, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 16: TND query/key/value之一维度不为3
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_qkv_wrong_dim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim, 1}, {tokenQ, numHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim, 1}, {tokenKv, numKvHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim, 1}, {tokenKv, numKvHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim, 1}, {tokenQ, numHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, headDim, 1}, {tokenQ, numHeads, headDim, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 17: BNSD query/key/value之一维度不为4
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bnsd_qkv_wrong_dim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim, 1}, {batch, numHeads, qSeqlen, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim, 1}, {batch, numKvHeads, kvSeqlen, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim, 1}, {batch, numKvHeads, kvSeqlen, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim, 1}, {batch, numHeads, qSeqlen, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1, 1}, {batch, numHeads, qSeqlen, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 18: BNSD query与key/value的batch维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bnsd_q_kv_batch_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch + 1, numKvHeads, kvSeqlen, headDim}, {batch + 1, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch + 1, numKvHeads, kvSeqlen, headDim}, {batch + 1, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 19: BNSD query与key/value的headDim维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bnsd_q_kv_headDim_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim + 1}, {batch, numKvHeads, kvSeqlen, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim + 1}, {batch, numKvHeads, kvSeqlen, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 20: BSND query/key/value之一维度不为4
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bsnd_qkv_wrong_dim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim, 1}, {batch, qSeqlen, numHeads, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim, 1}, {batch, kvSeqlen, numKvHeads, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim, 1}, {batch, kvSeqlen, numKvHeads, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 21: BSND query与key/value的batch维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bsnd_q_kv_batch_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch + 1, kvSeqlen, numKvHeads, headDim}, {batch + 1, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch + 1, kvSeqlen, numKvHeads, headDim}, {batch + 1, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 22: BSND query与key/value的headDim维度大小不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, bsnd_q_kv_headDim_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim + 1}, {batch, kvSeqlen, numKvHeads, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim + 1}, {batch, kvSeqlen, numKvHeads, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 23: A5 fp8场景不支持的AttentionOut的数据类型
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_unsupported_attentionOut_dtype)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1}, {batch, numHeads, qBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_INT4, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 24: TND场景未传入actualSeqLengths或actualSeqLengthsKv
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_actualSeqLengths_or_actualSeqLengthsKv_is_null)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 25: TND场景actualSeqLengths与actualSeqLengthsKv的长度不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, tnd_actualSeqLengths_actualSeqLengthsKv_size_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 26: 非TND场景actualSeqLengths与actualSeqLengthsKv的长度不一致
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_tnd_actualSeqLengths_actualSeqLengthsKv_size_not_equal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 27: 非TND场景actualSeqLengths中的最大值超过qSeqlen
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_tnd_actualSeqLengths_value_larger_than_maxQSeqlen)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t invalidActualSeqLengthsData[] = {qSeqlen + 1};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidActualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 28: 非TND场景actualSeqLengthsKv中的最大值超过kvSeqlen
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_tnd_actualSeqLengthsKv_value_larger_than_maxKvSeqlen)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t invalidActualSeqLengthsKvData[] = {kvSeqlen + 1};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidActualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 29: 非TND场景不传actualSeqLengths与actualSeqLengthsKv
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_tnd_no_actualSeqLengths_actualSeqLengthsKv)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9000000050000003);
}

// ============================================================================
// 用例 30: 非TND场景actualSeqLengths与actualSeqLengthsKv只传了其中之一
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, non_tnd_actualSeqLengths_actualSeqLengthsKv_not_both)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qSeqlen, 1}, {batch, numHeads, qSeqlen, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 31: blockShape中的值为负数
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockShape_contains_negative)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t invalidBlockShapeData[] = {-1, -1};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 32: blockShapeY的值不是16的整倍数
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockShapeY_not_multiple_of_16)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t invalidBlockShapeX = 128;
    constexpr int64_t invalidBlockShapeY = 129;
    int64_t invalidBlockShapeData[] = {invalidBlockShapeX, invalidBlockShapeY};
    constexpr int64_t invalidQBlockNum = (qSeqlen + invalidBlockShapeX - 1) / invalidBlockShapeX;
    constexpr int64_t invalidKvBlockNum = (kvSeqlen + invalidBlockShapeY - 1) / invalidBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, invalidQBlockNum, invalidKvBlockNum},
           {batch, numHeads, invalidQBlockNum, invalidKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 32b: A5 blockShapeX 可不对齐16；仅要求 blockShapeY 16 对齐
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_blockShapeX_not_multiple_of_16)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t unalignedBlockShapeX = 17;
    constexpr int64_t validBlockShapeY = 64;
    int64_t blockShapeDataLocal[] = {unalignedBlockShapeX, validBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + unalignedBlockShapeX - 1) / unalignedBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + validBlockShapeY - 1) / validBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeDataLocal},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400002);
}

// ============================================================================
// 用例 32c: A2 blockShapeX 可不对齐128；仅要求 blockShapeY 为128倍数
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A2_blockShape_not_multiple_of_128)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t unalignedBlockShapeX = 64;
    constexpr int64_t validBlockShapeY = 128;
    int64_t blockShapeDataLocal[] = {unalignedBlockShapeX, validBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + unalignedBlockShapeX - 1) / unalignedBlockShapeX;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, kvBlockNum}, {batch, numHeads, smallQBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeDataLocal},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ============================================================================
// 用例 32d: A5 16对齐小粒度blockShape正常场景
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_small_blockShape_16_64_tnd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t smallBlockShapeX = 16;
    constexpr int64_t smallBlockShapeY = 64;
    int64_t smallBlockShapeData[] = {smallBlockShapeX, smallBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + smallBlockShapeX - 1) / smallBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + smallBlockShapeY - 1) / smallBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, smallBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400002);
}

// ============================================================================
// 用例 32e: A5 blockShapeY>256 且非128整倍数（如528）允许通过；
// kvBaseTile 取 Y 的不大于 256 的最大约数（528 -> 176），避免整块 Y 入 L1 溢出
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_blockShapeY_gt256_not_multiple_of_128)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t fineBlockShapeX = 16;
    constexpr int64_t fineBlockShapeY = 528;
    int64_t fineBlockShapeData[] = {fineBlockShapeX, fineBlockShapeY};
    constexpr int64_t fineQBlockNum = (qSeqlen + fineBlockShapeX - 1) / fineBlockShapeX;
    constexpr int64_t fineKvBlockNum = (kvSeqlen + fineBlockShapeY - 1) / fineBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, fineQBlockNum, fineKvBlockNum}, {batch, numHeads, fineQBlockNum, fineKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, fineBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400002);
}

// ============================================================================
// 用例 33: blockShape的长度不为2
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockShape_wrong_size)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t invalidBlockShapeData[] = {128, 128, 128};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 34: blockSparseMask的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockSparseMask_wrong_dim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum, 1}, {batch, numHeads, qBlockNum, kvBlockNum, 1}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 35: blockSparseMask的某一维度大小不正确
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockSparseMask_one_of_dims_wrong)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch + 1, numHeads, qBlockNum, kvBlockNum}, {batch + 1, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 36: blockSparseMask的没有传
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockSparseMask_is_null)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 37: A5 fp8场景未传入qDequantScale/kDequantScale/vDequantScale
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_dequantScale_is_null)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 38: A5 fp8场景qDequantScale/kDequantScale/vDequantScale的数据类型不为fp32
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_dequantScale_wrong_dtype)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1}, {batch, numHeads, qBlockNum, 1}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 39: A5 fp8场景qDequantScale/kDequantScale/vDequantScale的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_dequantScale_wrong_dim)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1, 1}, {batch, numHeads, qBlockNum, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1, 1}, {batch, numKvHeads, kvBlockNum, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1, 1}, {batch, numKvHeads, kvBlockNum, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 40: A5 fp8场景qDequantScale/kDequantScale/vDequantScale的某一维度大小不正确
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_dequantScale_one_of_dims_wrong)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 2}, {batch, numHeads, qBlockNum, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 2}, {batch, numKvHeads, kvBlockNum, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 2}, {batch, numKvHeads, kvBlockNum, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 41: A5 非fp8场景qDequantScale/kDequantScale/vDequantScale不为空
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_non_fp8_dequantScale_not_null)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1}, {batch, numHeads, qBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 42: A5 fp8 场景 blockShapeX 可不对齐16；仅要求 blockShapeY 16 对齐
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_blockShapeX_is_not_128)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t unalignedBlockShapeX = 17;
    constexpr int64_t validBlockShapeY = 64;
    int64_t blockShapeDataLocal[] = {unalignedBlockShapeX, validBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + unalignedBlockShapeX - 1) / unalignedBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + validBlockShapeY - 1) / validBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeDataLocal},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, 1}, {batch, numHeads, smallQBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400012);
}

// ============================================================================
// 用例 43: A5 fp8场景blockShapeY不是16的整倍数
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_blockShapeX_is_not_multiple_of_256)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t validBlockShapeX = 16;
    constexpr int64_t invalidBlockShapeY = 130;
    int64_t invalidBlockShapeData[] = {validBlockShapeX, invalidBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + validBlockShapeX - 1) / validBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + invalidBlockShapeY - 1) / invalidBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, invalidBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, 1}, {batch, numHeads, smallQBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 43b: A5 fp8场景16对齐小粒度blockShape正常场景
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_small_blockShape_16_64_tnd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t smallBlockShapeX = 16;
    constexpr int64_t smallBlockShapeY = 64;
    int64_t smallBlockShapeData[] = {smallBlockShapeX, smallBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + smallBlockShapeX - 1) / smallBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + smallBlockShapeY - 1) / smallBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, smallBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, 1}, {batch, numHeads, smallQBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400012);
}

// ============================================================================
// 用例 43c: A5 fp8场景64x64细粒度blockShape正常场景
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_small_blockShape_64_64_tnd_normal)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t smallBlockShapeX = 64;
    constexpr int64_t smallBlockShapeY = 64;
    int64_t smallBlockShapeData[] = {smallBlockShapeX, smallBlockShapeY};
    constexpr int64_t smallQBlockNum = (qSeqlen + smallBlockShapeX - 1) / smallBlockShapeX;
    constexpr int64_t smallKvBlockNum = (kvSeqlen + smallBlockShapeY - 1) / smallBlockShapeY;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, smallKvBlockNum}, {batch, numHeads, smallQBlockNum, smallKvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, smallBlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, smallQBlockNum, 1}, {batch, numHeads, smallQBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, smallKvBlockNum, 1}, {batch, numKvHeads, smallKvBlockNum, 1}},
          ge::DT_FLOAT,
          ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400012);
}

// ============================================================================
// 用例 44: A5 fp8场景attentionOut的数据类型为fp16
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_attentionOut_dtype_fp16)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t fp8BlockShapeData[] = {128, 256};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, fp8BlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1}, {batch, numHeads, qBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400012);
}

// ============================================================================
// 用例 45: A5 fp8场景attentionOut的数据类型为bf16
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_fp8_attentionOut_dtype_bf16)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    int64_t fp8BlockShapeData[] = {128, 256};
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, fp8BlockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, 1}, {batch, numHeads, qBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvBlockNum, 1}, {batch, numKvHeads, kvBlockNum, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9050000030400022);
}

// ============================================================================
// 用例 46: numKeyValueHeads为0
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, numKeyValueHeads_is_0)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 47: A5 innerPrecise不为4
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A5_innerPrecise_is_not_4)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{batch, qSeqlen, numHeads, 1}, {batch, qSeqlen, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 48: A2A3 fp16场景innerPrecise不为0或1
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A2A3_fp16_innerPrecise_is_not_0_or_1)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 49: A2A3 bf16场景innerPrecise不为0
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, A2A3_bf16_innerPrecise_is_not_0)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 50: blockSize不为0
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, blockSize_is_not_0)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 51: preTokens/nextTokens不为2147483647
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, preTokens_nextTokens_is_not_2147483647)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 52: softmaxLseFlag不为0或1
// ============================================================================
TEST_F(block_sparse_attention_tiling_ut, softmaxLseFlag_is_not_0_or_1)
{
    struct BlockSparseAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqLengthsKvData},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenQ, numHeads, 1}, {tokenQ, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
        &compileInfo, "Ascend910B", 64, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
