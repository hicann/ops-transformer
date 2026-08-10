/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class MoeReRoutingV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeReRoutingV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeReRoutingV2Proto TearDown" << std::endl;
    }
};

TEST_F(MoeReRoutingV2, moe_re_routing_v2_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{256, 1}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 7168}, {256}, {256}, {16}, {256, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_scale)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{36, 385}, {36, 385}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{3, 12}, {3, 12}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{36, 1}, {36, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{36, 1}, {36, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{36, 385}, {36, 1}, {36}, {12}, {36, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1, 1}, {-1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-1, 1}, {-1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, 1}, {-1}, {-1}, {-1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_unknownshape)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2, 1}, {-2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-2, 1}, {-2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2, 1}, {-2}, {-2}, {-2, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_no_topk_weight)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 7168}, {256}, {256}, {16}, {0, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_inferdtype)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeReRoutingV2"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeReRoutingV2")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT32;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_FLOAT;
    ge::DataType output_2 = ge::DT_INT32;
    ge::DataType output_3 = ge::DT_INT32;
    ge::DataType output_4 = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 5)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({})
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1, &input_2, &input_3})
                              .OutputDataTypes({&output_0, &output_1, &output_2, &output_3, &output_4})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_tokens_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256, 1}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_infershape_expert_token_num_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRoutingV2",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256, 1}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeReRoutingV2, MoeReRoutingV2_inferdtype_no_topk_weight)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeReRoutingV2"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeReRoutingV2")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT32;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_FLOAT;
    ge::DataType output_2 = ge::DT_INT32;
    ge::DataType output_3 = ge::DT_INT32;
    ge::DataType output_4 = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(3, 5)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({})
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1, &input_2})
                              .OutputDataTypes({&output_0, &output_1, &output_2, &output_3, &output_4})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT);
}
