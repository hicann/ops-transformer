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
 * \file test_grouped_matmul_finalize_routing.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RecurrentGatedDeltaRuleTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "RecurrentGatedDeltaRuleTest Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "RecurrentGatedDeltaRuleTest Proto TearDown" << std::endl;
    }
};

TEST_F(RecurrentGatedDeltaRuleTest, Test0)
{
    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{b}, {b}};
    gert::StorageShape ssmStateIndicesShape = {{t}, {t}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};
    gert::StorageShape gkShape = {{}, {}};
    gert::StorageShape accTokensShape = {{b}, {b}};

    gert::InfershapeContextPara infershapeContextPara(
        "RecurrentGatedDeltaRule",
        {
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {keyShape, ge::DT_BF16, ge::FORMAT_ND},
            {valueShape, ge::DT_BF16, ge::FORMAT_ND},
            {betaShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_BF16, ge::FORMAT_ND},
            {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIndicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"sacle_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{t, nv, dv}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RecurrentGatedDeltaRuleTest, InferDataType_BF16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("RecurrentGatedDeltaRule");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType bf16Dtype = ge::DT_BF16;
    ge::DataType int32Dtype = ge::DT_INT32;
    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType outDtype = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("RecurrentGatedDeltaRule")
                             .NodeIoNum(8, 2)
                             .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&bf16Dtype, &bf16Dtype, &bf16Dtype, &bf16Dtype, &bf16Dtype, &int32Dtype,
                                              &int32Dtype, &floatDtype})
                             .OutputDataTypes({&outDtype, &outDtype})
                             .Build();

    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
}

TEST_F(RecurrentGatedDeltaRuleTest, InferDataType_FLOAT)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("RecurrentGatedDeltaRule");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType bf16Dtype = ge::DT_BF16;
    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType int32Dtype = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("RecurrentGatedDeltaRule")
                             .NodeIoNum(8, 2)
                             .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&bf16Dtype, &bf16Dtype, &bf16Dtype, &bf16Dtype, &floatDtype, &int32Dtype,
                                              &int32Dtype, &floatDtype})
                             .OutputDataTypes({&outDtype, &floatDtype})
                             .Build();

    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

TEST_F(RecurrentGatedDeltaRuleTest, InvalidStateDim)
{
    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int b = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{nv, dv, dk}, {nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{b}, {b}};
    gert::StorageShape ssmStateIndicesShape = {{t}, {t}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::InfershapeContextPara infershapeContextPara(
        "RecurrentGatedDeltaRule",
        {
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {keyShape, ge::DT_BF16, ge::FORMAT_ND},
            {valueShape, ge::DT_BF16, ge::FORMAT_ND},
            {betaShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_BF16, ge::FORMAT_ND},
            {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIndicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"sacle_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
