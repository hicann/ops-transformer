/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

// 测试类：专门测试 BlockSparseAttentionGrad 的 InferShape 逻辑
class BlockSparseAttentionGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BlockSparseAttentionGrad InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- BlockSparseAttentionGrad InferShape UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 下的 Shape 推导
// ============================================================================
TEST_F(BlockSparseAttentionGradInferShapeTest, infershape_bnsd_layout)
{
    int64_t b = 2, n = 8, n_kv = 4, s = 256, s_kv = 128, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttentionGrad",
        {// 0: dout [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: query [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: key [B, N_kv, S_kv, D]
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3: value [B, N_kv, S_kv, D]
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4: attentionOut (REQUIRED)
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5: softmaxLse (REQUIRED, Float32)
         {{{b, n, s}, {b, n, s}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 6: blockSparseMaskOptional
         {{{-1}, {-1}}, ge::DT_UINT8, ge::FORMAT_ND},
         // 7: attenMaskOptional
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 8: blockShapeOptional
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         // 9: actualSeqLengthsOptional
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         // 10: actualSeqLengthsKvOptional
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// 预期的输出列表占位 (dq, dk, dv)
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性占位 (对齐 OpDef 里的 7 个属性顺序)
         {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n_kv)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}});

    // 断言期待结果：dQ=Q, dK=K, dV=V
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, s, d},       // 0: dQ 输出
        {b, n_kv, s_kv, d}, // 1: dK 输出
        {b, n_kv, s_kv, d}  // 2: dV 输出
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 2: TND Layout 下的 Shape 推导 (补齐所有输入防止 GE 框架报错)
// ============================================================================
TEST_F(BlockSparseAttentionGradInferShapeTest, infershape_tnd_layout)
{
    int64_t total_q = 1024, total_kv = 2048, n = 8, n_kv = 2, d = 64;
    int64_t batch = 1;

    gert::InfershapeContextPara infershapeContextPara(
        "BlockSparseAttentionGrad",
        {
            // 补齐 11 个输入
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 0: dout
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 1: query
            {{{total_kv, n_kv, d}, {total_kv, n_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: key
            {{{total_kv, n_kv, d}, {total_kv, n_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 3: value
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 4: attentionOut (REQUIRED)
            {{{total_q, n}, {total_q, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 5: softmaxLse (REQUIRED)
            {{{-1}, {-1}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // 6: blockSparseMaskOptional
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},                             // 7: attenMaskOptional
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                                   // 8: blockShapeOptional
            {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},                           // 9: actualSeqLengthsOptional
            {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND} // 10: actualSeqLengthsKvOptional
        },
        {{{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n_kv)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {total_q, n, d},     // 0: dQ 输出
        {total_kv, n_kv, d}, // 1: dK 输出
        {total_kv, n_kv, d}  // 2: dV 输出
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
