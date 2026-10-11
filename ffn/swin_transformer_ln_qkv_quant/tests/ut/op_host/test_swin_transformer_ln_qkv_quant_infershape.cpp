/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SwinTransformerLnQkvQuantInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwinTransformerLnQkvQuantInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwinTransformerLnQkvQuantInferShapeTest TearDown" << std::endl;
    }
};

struct SwinTransformerLnQkvQuantInfo {
    gert::StorageShape& xShape;
    gert::StorageShape& gammaShape;
    gert::StorageShape& betaShape;
    gert::StorageShape& weightShape;
    gert::StorageShape& biasShape;
    gert::StorageShape& quant_scaleShape;
    gert::StorageShape& quant_offsetShape;
    gert::StorageShape& dequant_scaleShape;
    std::vector<int64_t> expectOutShape;

    ge::DataType xDtype;
    ge::DataType gammaDtype;
    ge::DataType betaDtype;
    ge::DataType weightDtype;
    ge::DataType biasDtype;
    ge::DataType quant_scaleDtype;
    ge::DataType quant_offsetDtype;
    ge::DataType dequant_scaleDtype;
    ge::DataType yDtype;

    int64_t headNum = 3;
    int64_t seqLength = 32;
    float epsilon = 0.00001;
    int64_t oriHeight = 56;
    int64_t oriWeight = 112;
    int64_t hWinSize = 7;
    int64_t wWinSize = 7;
    bool weightTranspose = true;
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const SwinTransformerLnQkvQuantInfo ioInfo, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape yStorageShape = {};
    std::vector<gert::StorageShape*> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("SwinTransformerLnQkvQuant")
                             .NodeIoNum(8, 1)
                             .NodeInputTd(0, ioInfo.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ioInfo.gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ioInfo.betaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ioInfo.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ioInfo.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ioInfo.quant_scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ioInfo.quant_offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, ioInfo.dequant_scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ioInfo.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)

                             .InputTensors({(gert::Tensor*)&ioInfo.xShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.gammaShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.betaShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.weightShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.biasShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.quant_scaleShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.quant_offsetShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.dequant_scaleShape})

                             .OutputShapes(ouputShapes)
                             .Attr("head_num", int64_t(ioInfo.headNum))
                             .Attr("seq_length", int64_t(ioInfo.seqLength))
                             .Attr("epsilon", int64_t(ioInfo.epsilon))
                             .Attr("ori_height", int64_t(ioInfo.oriHeight))
                             .Attr("ori_weight", int64_t(ioInfo.oriWeight))
                             .Attr("h_win_size", int64_t(ioInfo.hWinSize))
                             .Attr("w_win_size", int64_t(ioInfo.wWinSize))
                             .Attr("weight_transpose", int64_t(ioInfo.weightTranspose))
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("SwinTransformerLnQkvQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), ioInfo.expectOutShape);
}

TEST_F(SwinTransformerLnQkvQuantInfershape, swin_transformer_ln_qkv_quant_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SwinTransformerLnQkvQuant",
        {
            {{{1, 6272, 96}, {1, 6272, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{288, 96}, {288, 96}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"seq_length", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(0.00001)},
            {"ori_height", Ops::Transformer::AnyValue::CreateFrom<int64_t>(56)},
            {"ori_weight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(112)},
            {"h_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"w_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"weight_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 3, 49, 32}, {128, 3, 49, 32}, {128, 3, 49, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
