/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_gdn_gating_infershape.cpp
 * \brief Infershape unit tests for FusedGdnGating.
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FusedGdnGatingInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedGdnGatingInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedGdnGatingInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(FusedGdnGatingInfershapeTest, ShapeInference)
{
    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};

    gert::InfershapeContextPara infershapeContextPara(
        "FusedGdnGating",
        {
            {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {aShape, ge::DT_BF16, ge::FORMAT_ND},
            {bShape, ge::DT_BF16, ge::FORMAT_ND},
            {dtBiasShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedGdnGatingInfershapeTest, InferDataType_BF16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("FusedGdnGating");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType bf16Dtype = ge::DT_BF16;
    ge::DataType gOutDtype = ge::DT_FLOAT;
    ge::DataType betaOutDtype = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("FusedGdnGating")
                             .NodeIoNum(4, 2)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&floatDtype, &bf16Dtype, &bf16Dtype, &floatDtype})
                             .OutputDataTypes({&gOutDtype, &betaOutDtype})
                             .Build();

    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
}

TEST_F(FusedGdnGatingInfershapeTest, InferDataType_FP16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("FusedGdnGating");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType fp16Dtype = ge::DT_FLOAT16;
    ge::DataType gOutDtype = ge::DT_FLOAT;
    ge::DataType betaOutDtype = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("FusedGdnGating")
                             .NodeIoNum(4, 2)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&floatDtype, &fp16Dtype, &fp16Dtype, &floatDtype})
                             .OutputDataTypes({&gOutDtype, &betaOutDtype})
                             .Build();

    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}

TEST_F(FusedGdnGatingInfershapeTest, InvalidDim)
{
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{numHeads}, {numHeads}};
    gert::StorageShape bShape = {{numHeads}, {numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};

    gert::InfershapeContextPara infershapeContextPara(
        "FusedGdnGating",
        {
            {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {aShape, ge::DT_BF16, ge::FORMAT_ND},
            {bShape, ge::DT_BF16, ge::FORMAT_ND},
            {dtBiasShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
