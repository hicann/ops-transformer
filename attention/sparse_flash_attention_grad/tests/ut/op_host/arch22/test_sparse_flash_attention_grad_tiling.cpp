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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SparseFlashAttentionGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionGradTiling TearDown" << std::endl;
    }
};

namespace {
// key = ((((10 + atten) * 10 + hasRope) * 10 + isBsnd) * 10 + deterministic) * 10 + kvMerge
constexpr uint64_t kTilingKeyBsndAttenNoRope = 110100UL;   // sparse_mode=3, no rope, BSND, non-deter, non-kv-merge
constexpr uint64_t kTilingKeyBsndAttenWithRope = 111100UL; // sparse_mode=3, rope,    BSND, non-deter, non-kv-merge
constexpr uint64_t kTilingKeyTndAttenNoRope = 110000UL;    // sparse_mode=3, no rope, TND,  non-deter, non-kv-merge
} // namespace

// =================== A1: BSND + sparse_mode=3 + no rope, fp16 (正例) ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_A1_bsnd_fp16_norope)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_query
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // d_key
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // d_value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // d_query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}                              // d_key_rope
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kTilingKeyBsndAttenNoRope);
}

// =================== A2: BSND + sparse_mode=3 + no rope, bf16 (正例) ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_A2_bsnd_bf16_norope)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{2, 32, 16, 128}, {2, 32, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key              input1
            {{{2, 32, 1, 8}, {2, 32, 1, 8}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{2, 32, 16, 128}, {2, 32, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // d_out            input3
            {{{2, 32, 16, 128}, {2, 32, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // out              input4
            {{{2, 16, 32}, {2, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},          // softmax_max      input5
            {{{2, 16, 32}, {2, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},          // softmax_sum      input6
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input9
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {{{{2, 32, 16, 128}, {2, 32, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kTilingKeyBsndAttenNoRope);
}

// =================== A3: BSND + sparse_mode=3 + rope (正例) ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_A3_bsnd_bf16_rope)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input9
            {{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input10
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // key_rope         input11
        },
        {{{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kTilingKeyBsndAttenWithRope);
}

// =================== A4: TND + sparse_mode=3 + no rope (正例) ===================
// TND 下 actual_seq_lengths_query / actual_seq_lengths_kv 必须实例化（GetBaseShapeInfo 强校验）
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_A4_tnd_bf16_norope)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    int64_t actualSeqQList[] = {64};   // 单 batch，prefix-sum = t1
    int64_t actualSeqKvList[] = {128}; // 单 batch，prefix-sum = t2
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // query            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input2
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // d_out            input3
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // out              input4
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},          // softmax_max      input5
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},          // softmax_sum      input6
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input7
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQList},  // actual_seq_q     input8
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKvList}, // actual_seq_kv    input9
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                           // query_rope       input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                           // key_rope         input11
        },
        {{{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kTilingKeyTndAttenNoRope);
}

// =================== E1: 非法 layout (负例) - layout="BSH" 不在 TND/BSND 支持范围 ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E1_invalid_layout)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {{{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")}, // 非法
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E2: n2 != 1 (负例) - 仅支持 n2=1 ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E2_n2_not_one)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 4, 128}, {1, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 2, 128}, {1, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 2, 4}, {1, 64, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 4, 128}, {1, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 4, 128}, {1, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 4, 64}, {1, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 4, 64}, {1, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 2, 128}, {1, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {{{{1, 64, 4, 128}, {1, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 2, 128}, {1, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 2, 128}, {1, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E3: n1 不是 1/2/4/8/.../128 的 2 的幂 (负例) ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E3_n1_not_power_of_two)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 3, 128}, {1, 64, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 3, 128}, {1, 64, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 3, 128}, {1, 64, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 3, 64}, {1, 3, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 3, 64}, {1, 3, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {{{{1, 64, 3, 128}, {1, 64, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E4: head_dim(query) < head_dim(value) (负例) ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E4_dq_less_than_dv)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query            input0
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7, d2 > d
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}                              // key_rope         input11
        },
        {{{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E5: sparse_mode 非法 (负例) - 仅支持 0/3 ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E5_invalid_sparse_mode)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // softmax_sum      input6
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope         input11
        },
        {{{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // 非法
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E6: TND 缺失 actual_seq_lengths_query (负例) ===================
// TND 下 GetBaseShapeInfo 显式要求 actSeqQ / actSeqKV 非空，否则在 CheckBaseInput 阶段失败
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E6_tnd_missing_actseq)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_max      input5
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_sum      input6
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                      // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                      // actual_seq_kv    input9
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // query_rope       input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // key_rope         input11
        },
        {{{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E7: query 维数与 layout 字符串长度不一致 (负例) ===================
// query 是 3 维但 layout="BSND" (期望 4 维) → strlen(layout) != dimSize 直接失败
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E7_layout_dim_mismatch)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key              input1
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},         // sparse_indices   input2
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // d_out            input3
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},         // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},         // softmax_sum      input6
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value            input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_q     input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_kv    input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                       // query_rope       input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                       // key_rope         input11
        },
        {{{{64, 8, 128}, {64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")}, // 4 字符 vs 3 维
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// =================== E8: BSND S1=0 (负例) - kernel t1Idx/curS1 除零防护 ===================
TEST_F(SparseFlashAttentionGradTiling, SparseFlashAttentionGrad_910b_tiling_E8_bsnd_s1_zero)
{
    struct SparseFlashAttentionGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 0, 8, 128}, {1, 0, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query S1=0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{1, 0, 1, 4}, {1, 0, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},           // sparse_indices
            {{{1, 0, 8, 128}, {1, 0, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // d_out
            {{{1, 0, 8, 128}, {1, 0, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // out
            {{{1, 8, 0}, {1, 8, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // softmax_max
            {{{1, 8, 0}, {1, 8, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // softmax_sum
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // actual_seq_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope
        },
        {{{{1, 0, 8, 128}, {1, 0, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
