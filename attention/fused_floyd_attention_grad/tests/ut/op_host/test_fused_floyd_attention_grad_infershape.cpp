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
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FusedFloydAttentionGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedFloydAttentionGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedFloydAttentionGradTest TearDown" << std::endl;
    }
};

TEST_F(FusedFloydAttentionGradTest, fused_floyd_attention_grad_infer_dtype_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedFloydAttentionGrad",
        {
            // input info
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query
            {{{2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key1
            {{{2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // value1
            {{{2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key2
            {{{2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value2
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // dy
            {{{2, 1, 16, 1, 384}, {2, 1, 16, 1, 384}}, ge::DT_BOOL, ge::FORMAT_ND},          // atten_mask
            {{{2, 4, 16, 256, 8}, {2, 4, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},         // softmax_max
            {{{2, 4, 16, 256, 8}, {2, 4, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},         // softmax_sum
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attention_in
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 16, 256, 128}, {2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}, {2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedFloydAttentionGradTest, fused_floyd_attention_grad_infer_dtype_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedFloydAttentionGrad",
        {
            // input info
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query
            {{{2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key1
            {{{2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value1
            {{{2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key2
            {{{2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value2
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // dy
            {{{2, 1, 16, 1, 384}, {2, 1, 16, 1, 384}}, ge::DT_BOOL, ge::FORMAT_ND},       // atten_mask
            {{{2, 4, 16, 256, 8}, {2, 4, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_max
            {{{2, 4, 16, 256, 8}, {2, 4, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},      // softmax_sum
            {{{2, 4, 16, 256, 128}, {2, 4, 16, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // attention_in
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// attr
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 16, 256, 128}, {2, 4, 16, 384, 128}, {2, 4, 16, 384, 128}, {2, 4, 384, 256, 128}, {2, 4, 384, 256, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
