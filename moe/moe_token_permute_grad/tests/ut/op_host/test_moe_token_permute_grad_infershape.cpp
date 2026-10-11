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

class MoeTokenPermuteGrad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteGrad, infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteGrad",
        {// input info
         {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6144, 5120},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteGrad, infershape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteGrad",
        {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6144, 5120},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteGrad, infershape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteGrad",
        {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6144, 5120},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteGrad, infershape_inferdatatype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenPermuteGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_dtype = ge::DT_BF16;
    ge::DataType index_dtype = ge::DT_INT32;
    ge::DataType output_dtype = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_dtype, &index_dtype})
                              .OutputDataTypes({&output_dtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), input_dtype);
}
