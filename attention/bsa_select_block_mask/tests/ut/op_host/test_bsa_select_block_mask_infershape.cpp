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

// 测试类：专门测试 BSASelectBlockMask 的 InferShape 逻辑
class BSASelectBlockMaskInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BSASelectBlockMask InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- BSASelectBlockMask InferShape UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 下的 Shape 推导
// mask_out = [B, N, ceil(S / blockShapeX), ceil(S_kv / blockShapeY)]
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bnsd_layout)
{
    int64_t b = 2, n = 8, s = 256, s_kv = 128, d = 128;
    int64_t blockShapeX = 64;
    int64_t blockShapeY = 128;
    int64_t blockShapeArr[2] = {blockShapeX, blockShapeY};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 0: query [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [B, N, S_kv, D]
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape = [blockShapeX, blockShapeY]（shape 为 [2] 的常量张量）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeArr},
         // 3: post_block_shape (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 5: actual_seq_lengths_kv (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6: actual_block_len_query (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// 预期的输出列表占位 (block_sparse_mask_out)
         {{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性占位 (对齐 Op Def 里的 5 个属性顺序)
         {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    // 断言期待结果：mask_out = [B, N, ceil(S/blockShapeX), ceil(S_kv/blockShapeY)]
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, (s + blockShapeX - 1) / blockShapeX, (s_kv + blockShapeY - 1) / blockShapeY}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 2: TND Layout 下的 Shape 推导
// batch 从 actual_seq_lengths 推导，sqLen/skvLen 取批内最大序列长度（非 total）
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_tnd_layout)
{
    // 2 个 batch：query 长度 [600, 1024]，kv 长度 [2048, 1536]
    int64_t n = 8, d = 128;
    int64_t batch = 2;
    int64_t maxQSeqlen = 1024;
    int64_t maxKvSeqlen = 2048;
    int64_t totalQ = 600 + 1024;   // 1624，若误用 total 会得到 ceil(1624/256)=7
    int64_t totalKv = 2048 + 1536; // 3584，若误用 total 会得到 ceil(3584/256)=14
    int64_t blockShapeX = 256, blockShapeY = 256;
    int64_t blockShapeArr[2] = {blockShapeX, blockShapeY};
    int64_t seqLensQ[2] = {600, 1024};
    int64_t seqLensKv[2] = {2048, 1536};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 0: query [T, N, D]
         {{{totalQ, n, d}, {totalQ, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [T_kv, N, D]
         {{{totalKv, n, d}, {totalKv, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape = [blockShapeX, blockShapeY]（shape 为 [2] 的常量张量）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeArr},
         // 3: post_block_shape (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths [batch]（常量张量）
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, seqLensQ},
         // 5: actual_seq_lengths_kv [batch]（常量张量）
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, seqLensKv},
         // 6: actual_block_len_query (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    // TND 布局下：batchSize=actual_seq_lengths 元素个数，sqLen/skvLen=批内最大序列长度
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {batch, n, (maxQSeqlen + blockShapeX - 1) / blockShapeX, (maxKvSeqlen + blockShapeY - 1) / blockShapeY}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 3: BF16 数据类型
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bf16_dtype)
{
    int64_t b = 1, n = 4, s = 512, s_kv = 512, d = 128;
    int64_t blockShapeX = 256;
    int64_t blockShapeY = 256;
    int64_t blockShapeArr[2] = {blockShapeX, blockShapeY};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 0: query [B, N, S, D] BF16
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1: key [B, N, S_kv, D] BF16
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 2: block_shape = [blockShapeX, blockShapeY]（shape 为 [2] 的常量张量）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeArr},
         // 3: post_block_shape (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 5: actual_seq_lengths_kv (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6: actual_block_len_query (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL，不传入)
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, (s + blockShapeX - 1) / blockShapeX, (s_kv + blockShapeY - 1) / blockShapeY}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 不合理场景用例：layout 不合法（仅支持 BNSD/TND，且 Q 与 KV 必须一致）
// ============================================================================

// q_input_layout 为不支持的字符串 "BSND"（仅支持 BNSD/TND），预期 InferShape 返回失败
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_invalid_q_layout)
{
    int64_t b = 2, n = 8, s = 256, s_kv = 128, d = 128;
    int64_t blockShapeArr[2] = {64, 128};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {{{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeArr},
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
