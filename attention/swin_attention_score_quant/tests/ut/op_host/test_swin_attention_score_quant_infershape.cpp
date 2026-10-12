/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_case_executor.h"

class SwinAttentionScoreQuantTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTest TearDown" << std::endl;
    }
};

namespace {

gert::InfershapeContextPara BuildInfershapePara(int64_t b, int64_t n, int64_t s, int64_t h, bool mask1_present = true)
{
    return gert::InfershapeContextPara(
        "SwinAttentionScoreQuant",
        {
            {{{b, n, s, h}, {b, n, s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{b, n, s, h}, {b, n, s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{b, n, s, h}, {b, n, s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1, h}, {1, h}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, h}, {1, h}}, ge::DT_INT32, ge::FORMAT_ND},
            mask1_present ? gert::InfershapeContextPara::TensorDescription({{1, n, s, s}, {1, n, s, s}}, ge::DT_FLOAT16,
                                                                           ge::FORMAT_ND) :
                            gert::InfershapeContextPara::TensorDescription({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND),
            {{{1, n, s, s}, {1, n, s, s}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"query_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        });
}

} // namespace

TEST_F(SwinAttentionScoreQuantTest, swin_attention_score_quant_infershape_with_mask_001)
{
    auto para = BuildInfershapePara(10, 3, 49, 32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{10, 3, 49, 32}});
}

TEST_F(SwinAttentionScoreQuantTest, swin_attention_score_quant_infershape_without_mask_002)
{
    auto para = BuildInfershapePara(10, 3, 49, 32, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{10, 3, 49, 32}});
}
