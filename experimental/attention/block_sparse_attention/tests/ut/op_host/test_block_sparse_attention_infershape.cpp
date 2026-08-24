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

class block_sparse_attention_infershape_ut : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "block_sparse_attention_infershape_ut SetUp" << endl; }
    static void TearDownTestCase() { cout << "block_sparse_attention_infershape_ut TearDown" << endl; }
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
constexpr int64_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;
constexpr int64_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;
constexpr int64_t tokenQ = batch * qSeqlen;
constexpr int64_t tokenKv = batch * kvSeqlen;
constexpr float scaleValue = 1.0 / sqrt(static_cast<float>(headDim));
} // namespace

// ============================================================================
// 用例 1: TND正常场景
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, tnd_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 2: BNSD正常场景
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 3: BSND正常场景
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 4: TND query的维度不为3
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, tnd_q_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim, 1}, {tokenQ, numHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 5: BNSD query的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_q_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim, 1}, {batch, numHeads, qSeqlen, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 6: BSND query的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_q_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim, 1}, {batch, qSeqlen, numHeads, headDim, 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 7: 不支持的qInputLayout
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, unsupported_qInputLayout)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 8: qInputLayout和kvInputLayout不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, qInputLayout_kvInputLayout_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 9: TND key/value的维度不为3
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, tnd_kv_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim, 1}, {tokenKv, numKvHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim, 1}, {tokenKv, numKvHeads, headDim, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 10: TND key/value numKvHeads维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, tnd_kv_numHeads_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads + 1)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 11: TND key/value headDim维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, tnd_kv_headDim_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{tokenQ, numHeads, headDim}, {tokenQ, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim}, {tokenKv, numKvHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{tokenKv, numKvHeads, headDim + 1}, {tokenKv, numKvHeads, headDim + 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{tokenQ, numHeads, headDim}, {tokenQ, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 12: BNSD key/value的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_kv_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 13: BNSD key/value numKvHeads维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_kv_numHeads_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads + 1)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 14: BNSD key/value batch维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_kv_batch_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch + 1, numKvHeads, kvSeqlen, headDim}, {batch + 1, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 15: BNSD key/value headDim维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bnsd_kv_headDim_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, numHeads, qSeqlen, headDim}, {batch, numHeads, qSeqlen, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim}, {batch, numKvHeads, kvSeqlen, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numKvHeads, kvSeqlen, headDim + 1}, {batch, numKvHeads, kvSeqlen, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, numHeads, qSeqlen, headDim},
                                                           {batch, numHeads, qSeqlen, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 16: BSND key/value的维度不为4
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_kv_wrong_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 17: BSND key/value numKvHeads维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_kv_numHeads_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
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
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads + 1)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 18: BSND key/value batch维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_kv_batch_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch + 1, kvSeqlen, numKvHeads, headDim}, {batch + 1, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 19: BSND key/value headDim维度不一致
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, bsnd_kv_headDim_not_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{batch, qSeqlen, numHeads, headDim}, {batch, qSeqlen, numHeads, headDim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim}, {batch, kvSeqlen, numKvHeads, headDim}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, kvSeqlen, numKvHeads, headDim + 1}, {batch, kvSeqlen, numKvHeads, headDim + 1}},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{batch, qSeqlen, numHeads, headDim},
                                                           {batch, qSeqlen, numHeads, 1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ============================================================================
// 用例 20: UNKNOWN DIM场景
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, unknown_shape)
{
    constexpr int64_t UNKNOWN_DIMS = -2;
    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttention",
        {{{{UNKNOWN_DIMS}, {UNKNOWN_DIMS}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{UNKNOWN_DIMS}, {UNKNOWN_DIMS}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{UNKNOWN_DIMS}, {UNKNOWN_DIMS}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch, numHeads, qBlockNum, kvBlockNum}, {batch, numHeads, qBlockNum, kvBlockNum}},
          ge::DT_INT8,
          ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
         {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{UNKNOWN_DIMS}, {UNKNOWN_DIMS}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 用例 21: InferDataType
// ============================================================================
TEST_F(block_sparse_attention_infershape_ut, inferDataType)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("BlockSparseAttention")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType queryType = ge::DT_FLOAT16;
    ge::DataType keyType = ge::DT_FLOAT16;
    ge::DataType valueType = ge::DT_FLOAT16;
    ge::DataType blockSparseMaskType = ge::DT_INT8;
    ge::DataType attenMaskType = ge::DT_INT8;
    ge::DataType blockShapeType = ge::DT_INT64;
    ge::DataType actualSeqLengthsType = ge::DT_INT64;
    ge::DataType actualSeqLengthsKvType = ge::DT_INT64;
    ge::DataType blockTableType = ge::DT_INT32;
    ge::DataType qDequantScaleType = ge::DT_FLOAT;
    ge::DataType kDequantScaleType = ge::DT_FLOAT;
    ge::DataType vDequantScaleType = ge::DT_FLOAT;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(12, 2)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&queryType, &keyType, &valueType, &blockSparseMaskType, &attenMaskType, &blockShapeType,
                             &actualSeqLengthsType, &actualSeqLengthsKvType, &blockTableType, &qDequantScaleType,
                             &kDequantScaleType, &vDequantScaleType})
            .NodeAttrs({{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
                        {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
                        {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
                        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
                        {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
