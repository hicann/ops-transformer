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
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeGatingTopKSoftmaxV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_illegal_input_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_illegal_k_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_illegal_renorm_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2}, {-1, 2}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_softmax_output_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}, {4, 8}, {4, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_unknown_rank)
{
    // 未知秩：x = (-2,)，三个输出透传未知秩
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infer_datatype_01)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType expertIdxType = ge::DT_UNDEFINED;
    ge::DataType softmaxResultType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(1, 3)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType})
                             .OutputDataTypes({&yType, &expertIdxType, &softmaxResultType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeGatingTopKSoftmaxV2")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infer_datatype_02)
{
    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType expertIdxType = ge::DT_UNDEFINED;
    ge::DataType softmaxResultType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(1, 3)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType})
                             .OutputDataTypes({&yType, &expertIdxType, &softmaxResultType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeGatingTopKSoftmaxV2")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}
