/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fused_weight_infershape.cpp
 * \brief EngramFusedWeight InferShape / InferDataType UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class EngramFusedWeightProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFusedWeightProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFusedWeightProto TearDown" << std::endl;
    }
};

TEST_F(EngramFusedWeightProto, engram_fused_weight_infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "EngramFusedWeight",
        {
            {{{4, 5120}, {4, 5120}}, ge::DT_BF16, ge::FORMAT_ND}, // weight_hidden
            {{{4, 5120}, {4, 5120}}, ge::DT_BF16, ge::FORMAT_ND}, // weight_embed
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weight_fused (输出 shape 应与输入一致)
        },
        {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(EngramFusedWeightProto, engram_fused_weight_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("EngramFusedWeight")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputWhDtype = ge::DT_BF16;
    ge::DataType inputWeDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputWhDtype, &inputWeDtype})
                              .OutputDataTypes({&outDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
}
