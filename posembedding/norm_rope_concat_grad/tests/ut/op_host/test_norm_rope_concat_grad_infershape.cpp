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

class NormRopeConcatGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NormRopeConcatGradInfershape Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NormRopeConcatGradInfershape Proto Test TearDown" << std::endl;
    }
};

TEST_F(NormRopeConcatGradInfershape, NormRopeConcatGrad_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 16, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NormRopeConcatGradInfershape, NormRopeConcatGrad_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 16, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NormRopeConcatGradInfershape, NormRopeConcatGrad_infer_shape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 8, 32, 64},
                                                           {2, 16, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {2, 32, 32, 64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64},
                                                           {64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
