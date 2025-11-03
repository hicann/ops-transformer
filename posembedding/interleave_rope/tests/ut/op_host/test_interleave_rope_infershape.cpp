/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

class InterleaveRope : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InterleaveRope SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InterleaveRope TearDown" << std::endl;
    }
};

TEST_F(InterleaveRope, InterleaveRope_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InterleaveRope",
        {
            {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 32, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InterleaveRope, InterleaveRope_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InterleaveRope",
        {
            {{{32, 32, 4, 64}, {32, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 4, 64}, {32, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 4, 64}, {32, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 32, 4, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InterleaveRope, InterleaveRope_inferdtype)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("InterleaveRope"),
              nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                              .GetSpaceRegistry()
                              ->GetOpImpl("InterleaveRope")
                              ->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_0, &input_0})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
