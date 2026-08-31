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
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

class SparseFlashAttentionGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionGradProto TearDown" << std::endl;
    }
};

// 输入顺序（与 def.cpp 一致，sinks 插在 softmax_sum 后=input7）：
//   0 query, 1 key, 2 sparse_indices, 3 d_out, 4 out, 5 softmax_max, 6 softmax_sum,
//   7 sinks(OPTIONAL), 8 value, 9 act_seq_q, 10 act_seq_kv, 11 query_rope, 12 key_rope
// 输出：0 d_query, 1 d_key, 2 d_value, 3 d_query_rope, 4 d_key_rope, 5 d_sinks(OPTIONAL)

// =================== BSND 正例: 无 rope 无 sink, dq/dk/dv；d_sinks=[0]（空，跳过）===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_bsnd_norope)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_sum      input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sinks            input7 OPTIONAL（无 sink）
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope       input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_query
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_key
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_value
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_query_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_key_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 8, 128}, {1, 128, 1, 128}, {1, 128, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== BSND 正例: value 可选为空, 只推 dq/dk；d_sinks=[0] ==================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_value_optional_null)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_sum      input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // sinks            input7 OPTIONAL
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // value            input8 optional
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope       input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                              // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 8, 128}, {1, 128, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== BSND 正例: 带 rope, 推 dq/dk/dv/dq_rope/dk_rope；d_sinks=[0] =========
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_bsnd_rope)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_sum      input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // sinks            input7 OPTIONAL
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input10
            {{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input11
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 8, 128}, {1, 128, 1, 128}, {1, 128, 1, 128}, {1, 64, 8, 64}, {1, 128, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== TND 正例 (小写 layout, 内部 toupper)；d_sinks=[0] ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_tnd_lowercase)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_max      input5
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_sum      input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // sinks            input7 OPTIONAL
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input8
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                    // actual_seq_q     input9
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                    // actual_seq_kv    input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // query_rope       input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("tnd")}, // 小写 -> 内部 toupper
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 8, 128}, {128, 1, 128}, {128, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== sink 正例：TND + sinks 在 → d_sinks = sinks shape [N1] =========
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_tnd_with_sinks)
{
    constexpr int64_t n1 = 8;
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, n1, 128}, {64, n1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{64, n1, 128}, {64, n1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // d_out            input3
            {{{64, n1, 128}, {64, n1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // out              input4
            {{{1, 64, n1}, {1, 64, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max      input5
            {{{1, 64, n1}, {1, 64, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_sum      input6
            {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // sinks            input7（在！[N1]）
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input8
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                    // actual_seq_q     input9
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                    // actual_seq_kv    input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // query_rope       input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                        // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_query
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_key
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_value
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_query_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // d_key_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks（应 = [N1]）
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0846f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    // 输出顺序: dq, dk, dv, dq_rope, dk_rope, d_sinks
    // d_sinks = sinks shape = [N1]；d_query_rope/d_key_rope 空（无 rope）
    std::vector<std::vector<int64_t>> expectOutputShape = {{64, n1, 128}, {128, 1, 128}, {128, 1, 128}, {}, {}, {n1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== 负例: 非法 layout "BSH" ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_invalid_layout)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query            input0
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices   input2
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // d_out            input3
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // out              input4
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_max      input5
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // softmax_sum      input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // sinks            input7 OPTIONAL
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_q     input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // actual_seq_kv    input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope       input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // key_rope         input12
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // d_sinks
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
