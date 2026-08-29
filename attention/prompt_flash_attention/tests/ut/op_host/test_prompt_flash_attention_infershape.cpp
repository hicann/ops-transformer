/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PromptFlashAttentionProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "PromptFlashAttentionProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "PromptFlashAttentionProto TearDown" << std::endl;
    }
};

// BNSD
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_0)
{
    int64_t actual_seq_qlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                  538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    int64_t actual_seq_kvlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                   538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // query input0
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // key input1
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
            {{{23, 1024, 9088}, {23, 1024, 9088}}, ge::DT_BOOL, ge::FORMAT_ND},       // atten_mask input4
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
            {{{40, 128}, {40, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {23, 40, 1024, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// BSH
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_1)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},          // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},   // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},  // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {256, 14, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// BSND
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_2)
{
    int64_t actual_seq_qlist[] = {2048, 2048, 1024, 1024, 2048, 2028, 2048, 1024};
    int64_t actual_seq_kvlist[] = {2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // pse_shift input3
            {{{8, 1, 2048, 2048}, {8, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // quant_scale2 input8
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}        // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1000)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {8, 2048, 40, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// TND
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_3)
{
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},     // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {16384, 64, 192}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// BNSD_BSND
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_4)
{
    int64_t actual_seq_qlist[] = {37, 42, 2, 63, 6, 56, 40, 42};
    int64_t actual_seq_kvlist[] = {11, 11, 11, 11, 11, 11, 11, 11};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{8, 20, 65, 48}, {8, 20, 65, 48}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
            {{{8, 20, 65, 48}, {8, 20, 65, 48}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
            {{{8, 20, 65, 48}, {8, 20, 65, 48}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},        // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.1443375672974065f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {8, 65, 20, 48}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// numhead = 0
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_5)
{
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 0, 2048, 128}, {1, 0, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 0, 2048, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// NTD_TND FAILED
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_6)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 5, 2048, 128}, {1, 0, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD_TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 5, 2048, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// NTD_TND SUCCESS
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_7)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                              // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                             // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},   // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},  // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD_TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {14, 256, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// SH
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_8)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 5, 2048, 128}, {1, 0, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 5, 2048, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// q/k/v = -2
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_9)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // query input0
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key input1
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                            // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// NSD
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_10)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 5, 2048, 128}, {1, 0, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 5, 2048, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// TND_NTD success
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_11)
{
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},     // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {64, 16384, 192}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// TND_NTD failed
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_12)
{
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {64, 16384, 192}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// NTD FAILED
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_13)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 5, 2048, 128}, {1, 0, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 5, 2048, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// NTD FAILED
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_14)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{1, 5, 2048}, {1, 0, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // query input0
            {{{1, 5, 2048}, {1, 5, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // key input1
            {{{1, 5, 2048}, {1, 5, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                            // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 5, 2048}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// BNSD_BSND query dimNum != 4
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_15)
{
    int64_t actual_seq_qlist[] = {37, 42, 2, 63, 6, 56, 40, 42};
    int64_t actual_seq_kvlist[] = {11, 11, 11, 11, 11, 11, 11, 11};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{8, 20, 65}, {8, 20, 65}}, ge::DT_BF16, ge::FORMAT_ND},           // query input0
            {{{8, 20, 65}, {8, 20, 65}}, ge::DT_BF16, ge::FORMAT_ND},           // key input1
            {{{8, 20, 65}, {8, 20, 65}}, ge::DT_BF16, ge::FORMAT_ND},           // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},        // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.1443375672974065f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {8, 65, 20, 48}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// TND qv dimNum != 3
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_16)
{
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{16384, 64, 192, 1}, {16384, 64, 192, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {16384, 64, 192}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// BSH q dimNum != 3
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_17)
{
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{256, 14, 5120, 1}, {256, 14, 5120, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{256, 14, 5120, 1}, {256, 14, 5120, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{256, 14, 5120, 1}, {256, 14, 5120, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},                // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},         // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},        // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                    // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {256, 14, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// BNSD query dimNum != 4
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_18)
{
    int64_t actual_seq_qlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                  538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    int64_t actual_seq_kvlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                   538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{23, 40, 1024}, {23, 40, 1024}}, ge::DT_INT8, ge::FORMAT_ND},       // query input0
            {{{23, 40, 9088}, {23, 40, 9088}}, ge::DT_INT8, ge::FORMAT_ND},       // key input1
            {{{23, 40, 9088}, {23, 40, 9088}}, ge::DT_INT8, ge::FORMAT_ND},       // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
            {{{23, 1024, 9088}, {23, 1024, 9088}}, ge::DT_BOOL, ge::FORMAT_ND},   // atten_mask input4
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{40, 128}, {40, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {23, 40, 1024, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// BSND query dimNum != 4
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_infershape_19)
{
    int64_t actual_seq_qlist[] = {2048, 2048, 1024, 1024, 2048, 2028, 2048, 1024};
    int64_t actual_seq_kvlist[] = {2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024};
    gert::InfershapeContextPara infershapeContextPara(
        "PromptFlashAttention",
        // 输入Tensor
        {
            {{{8, 2048, 40}, {8, 2048, 40}}, ge::DT_BF16, ge::FORMAT_ND},           // query input0
            {{{8, 2048, 40}, {8, 2048, 40}}, ge::DT_BF16, ge::FORMAT_ND},           // key input1
            {{{8, 2048, 40}, {8, 2048, 40}}, ge::DT_BF16, ge::FORMAT_ND},           // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // pse_shift input3
            {{{8, 1, 2048, 2048}, {8, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // quant_scale2 input8
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}        // quant_offset2 input9
        },
        {// 输出Tensor
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1000)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {8, 2048, 40, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 比对成功返回SUCCESS
}

// infer dataType
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_inferdtype_0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("PromptFlashAttention")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_INT8;
        ge::DataType input_ref1 = ge::DT_FLOAT16;
        ge::DataType input_ref2 = ge::DT_BOOL;
        ge::DataType input_ref3 = ge::DT_INT64;
        ge::DataType input_ref4 = ge::DT_UINT64;
        ge::DataType input_ref5 = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(12, 1)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref1, &input_ref2, &input_ref3, &input_ref3,
                                 &input_ref4, &input_ref5, &input_ref4, &input_ref5, &input_ref5})
                .NodeAttrs({{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
                            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
                            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
                            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
                            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

// infer dataType
TEST_F(PromptFlashAttentionProto, PromptFlashAttention_inferdtype_1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("PromptFlashAttention")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_INT8;
        ge::DataType input_ref1 = ge::DT_FLOAT16;
        ge::DataType input_ref2 = ge::DT_BOOL;
        ge::DataType input_ref3 = ge::DT_INT64;
        ge::DataType input_ref4 = ge::DT_UINT64;
        ge::DataType input_ref5 = ge::DT_FLOAT;
        ge::DataType input_ref6 = ge::DT_UNDEFINED;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(12, 1)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref1, &input_ref2, &input_ref3, &input_ref3,
                                 &input_ref4, &input_ref5, &input_ref4, &input_ref6, &input_ref5})
                .NodeAttrs({{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
                            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
                            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
                            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
                            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
