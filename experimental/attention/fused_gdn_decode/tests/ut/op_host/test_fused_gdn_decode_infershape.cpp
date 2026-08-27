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

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

TEST(FusedGdnDecodeInferShapeTest, InfersOutputAndState)
{
    constexpr int64_t batch = 4;
    constexpr int64_t qkHeads = 8;
    constexpr int64_t valueHeads = 16;
    constexpr int64_t keyDim = 128;
    constexpr int64_t valueDim = 128;
    constexpr int64_t stateSlots = 32;
    constexpr int64_t mixedDim = 2 * qkHeads * keyDim + valueHeads * valueDim;

    gert::StorageShape mixedShape = {{batch, mixedDim}, {batch, mixedDim}};
    gert::StorageShape gateShape = {{batch, valueHeads}, {batch, valueHeads}};
    gert::StorageShape paramShape = {{valueHeads}, {valueHeads}};
    gert::StorageShape stateShape = {{stateSlots, valueHeads, valueDim, keyDim},
                                     {stateSlots, valueHeads, valueDim, keyDim}};
    gert::StorageShape indexShape = {{batch}, {batch}};
    gert::InfershapeContextPara context(
        "FusedGdnDecode",
        {
            {mixedShape, ge::DT_BF16, ge::FORMAT_ND},
            {gateShape, ge::DT_BF16, ge::FORMAT_ND},
            {gateShape, ge::DT_BF16, ge::FORMAT_ND},
            {paramShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {paramShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {indexShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838835f)},
            {"softplus_threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
        });

    std::vector<std::vector<int64_t>> expected = {
        {batch, 1, valueHeads, valueDim},
        {stateSlots, valueHeads, valueDim, keyDim},
    };
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST(FusedGdnDecodeInferShapeTest, InfersIndependentStateDtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto opImpl = registry->GetOpImpl("FusedGdnDecode");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    ge::DataType bf16 = ge::DT_BF16;
    ge::DataType fp32 = ge::DT_FLOAT;
    ge::DataType int32 = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_UNDEFINED;
    ge::DataType stateOutDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("FusedGdnDecode")
                      .NodeIoNum(7, 2)
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&bf16, &bf16, &bf16, &fp32, &bf16, &fp32, &int32})
                      .OutputDataTypes({&outDtype, &stateOutDtype})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}
