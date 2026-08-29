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

class IncreFlashAttentionProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "IncreFlashAttentionProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "IncreFlashAttentionProto TearDown" << std::endl;
    }
};

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 1024}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2:v:
            {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 24, 128}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 1, 21}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 1, 21}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 1, 21}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 1, 21}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 1024}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 1024}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(IncreFlashAttentionProto, IncreFlashAttention_950_infershape_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IncreFlashAttention",
        // 输入Tensor
        {
            // 0:q
            {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1:k:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2:v:
            {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3:pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4:atten_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 5:actual_seq_lengths
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6:dequant_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 7:quant_scale1
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 8:dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 9:quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 10:quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 11:antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 12:antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 13:block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // 14:kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor

         {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 1024}, // attentionOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
