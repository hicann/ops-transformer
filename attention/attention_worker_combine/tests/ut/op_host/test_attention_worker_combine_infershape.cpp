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

class AttentionWorkerCombineInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionWorkerCombineInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionWorkerCombineInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(AttentionWorkerCombineInfershapeTest, AttentionWorkerCombine_infershape_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionWorkerCombine",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
         {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
        {// output
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 7168}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
TEST_F(AttentionWorkerCombineInfershapeTest, mxfp8_output_shape)
{
    for (int64_t dtype : {2, 3, 4}) {
        gert::InfershapeContextPara para(
            "AttentionWorkerCombine",
            {{{{1024}, {1024}}, ge::DT_INT8, ge::FORMAT_ND},
             {{{3, 2, 4}, {3, 2, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
            {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
            {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65)},
             {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype)},
             {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 65}, {1}});
    }
}

TEST_F(AttentionWorkerCombineInfershapeTest, token_dtype_output_type)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("AttentionWorkerCombine");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (int64_t dtype : {0, 1, 2, 3, 4, 5}) {
        ge::DataType ctxType = ge::DT_INT8;
        ge::DataType scaleType = dtype >= 2 ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT;
        ge::DataType layerType = ge::DT_INT32;
        ge::DataType yType = ge::DT_UNDEFINED;
        ge::DataType nextType = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .SetOpType("AttentionWorkerCombine")
                          .IrInputNum(3)
                          .NodeIoNum(3, 2)
                          .InputDataTypes({&ctxType, &scaleType, &layerType})
                          .OutputDataTypes({&yType, &nextType})
                          .NodeAttrs({{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65)},
                                      {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype)},
                                      {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(impl->infer_datatype(context), dtype == 5 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS);
        if (dtype != 5) {
            EXPECT_EQ(context->GetOutputDataType(0), dtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16);
            EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
        }
    }
}
