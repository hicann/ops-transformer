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

class MoeTokenPermuteWithEpInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_case_0)
{
    std::vector<int64_t> range({1, 5});
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_1d_indices)
{
    std::vector<int64_t> range({0, 4});
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEp",
        {{{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}, {4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_indices_dim_invalid)
{
    std::vector<int64_t> range({0, 6});
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_unknownshape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEp",
        {{{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 4})},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_range_start_ge_end)
{
    std::vector<int64_t> range({5, 1});
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto* opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteWithEp");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ge::DataType inputTokens = ge::DT_BF16;
    ge::DataType inputIndices = ge::DT_INT64;
    ge::DataType inputProbs = ge::DT_BF16;
    ge::DataType outputTokens = ge::DT_UNDEFINED;
    ge::DataType outputIndices = ge::DT_UNDEFINED;
    ge::DataType outputProbs = ge::DT_UNDEFINED;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(3)
                             .NodeIoNum(3, 3)
                             .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&inputTokens, &inputIndices, &inputProbs})
                             .OutputDataTypes({&outputTokens, &outputIndices, &outputProbs})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
}
