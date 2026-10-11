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

class MoeGatingTopKSoftmax : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmax Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmax Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_input)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
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
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_input_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_input_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_input_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
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
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
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
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmax",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// inferDtype
TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infer_datatype_01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeGatingTopKSoftmax")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType indices_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref})
                                  .OutputDataTypes({&output_ref, &indices_ref, &indices_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(2), indices_ref);
    }
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infer_datatype_02)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeGatingTopKSoftmax")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        ge::DataType indices_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref})
                                  .OutputDataTypes({&output_ref, &indices_ref, &indices_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(2), indices_ref);
    }
}
