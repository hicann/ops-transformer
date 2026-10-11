/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moeTokenUnpermuteGrad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermuteGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteGradInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteGradInferShape TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 64}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_case_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 64}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_prob_not_none_split_h_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 8192}, {10, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 8192}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_prob_not_none_split_h_fp16_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 8192}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_prob_not_none_split_h_fp16_case_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 8192}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_inferdtype_bf16_case_0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermuteGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_inferdtype_bf16_case_1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermuteGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_ref_prob = ge::DT_FLOAT;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_BF16;
        ge::DataType output_ref_prob = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_indices_ref, &input_ref_prob})
                                  .OutputDataTypes({&output_ref, &output_ref_prob})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref_prob);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_prob);
    }
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_inferdtype_fp16_case_0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermuteGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_prob_none_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 64}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               30,
           },
           {
               30,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_invalid_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenUnpermuteGrad",
        {{{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{30, 1}, {30, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_inferdtype_fp32_case_0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermuteGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}
