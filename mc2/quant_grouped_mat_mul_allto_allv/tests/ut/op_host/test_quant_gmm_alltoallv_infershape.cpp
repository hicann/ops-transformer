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
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace QuantGroupedMatMulAlltoAllvUT {
struct TilingParams {
    int64_t BSK{4096};
    int64_t BS{2048};
    int64_t H1{7168};
    int64_t H2{7168};
    int64_t A{4096};
    int64_t N1{4096};
    int64_t N2{4096};
    int64_t epWorldSize{8};
    int64_t e{4};
    int64_t gmmWeightDim1{7168};
    int64_t gmmWeightDim2{4096};
    int64_t mmWeightDim0{7168};
    int64_t mmWeightDim1{4096};
    bool transGmmWeight{false};
    bool transMmWeight{false};
    bool hasMmInput{true};
    int64_t yDtype{28};
    int64_t mmDtype{28};
    std::string group{"group"};
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

class QuantGroupedMatMulAlltoAllvInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGroupedMatMulAlltoAllvInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGroupedMatMulAlltoAllvInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferDataTypeTest_Normal)
{
    int64_t inputNum{6};
    int64_t outputNum{2};
    auto tilingParam = TilingParams{};

    std::vector<ge::DataType> inputDtypes{ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_INT64,
                                          ge::DT_INT64,    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8};

    std::vector<void*> inputDtypesPtrs(inputNum);
    for (int64_t i = 0; i < inputNum; i++) {
        inputDtypesPtrs[i] = &inputDtypes[i];
    }
    std::vector<void*> outputDtypesPtrs(outputNum);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(inputNum, outputNum)
            .InputDataTypes(inputDtypesPtrs)
            .OutputDataTypes(outputDtypesPtrs)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParam.group)},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.epWorldSize)},
                 {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.sendCounts)},
                 {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.recvCounts)},
                 {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transGmmWeight)},
                 {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transMmWeight)},
                 {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.yDtype)},
                 {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.mmDtype)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantGroupedMatMulAlltoAllv")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_HIFLOAT8);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_HIFLOAT8);
}

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferDataTypeTest_Float8E5M2)
{
    int64_t inputNum{6};
    int64_t outputNum{2};
    auto tilingParam = TilingParams{};

    std::vector<ge::DataType> inputDtypes{ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_INT64,
                                          ge::DT_INT64,       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};

    std::vector<void*> inputDtypesPtrs(inputNum);
    for (int64_t i = 0; i < inputNum; i++) {
        inputDtypesPtrs[i] = &inputDtypes[i];
    }
    std::vector<void*> outputDtypesPtrs(outputNum);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(inputNum, outputNum)
            .InputDataTypes(inputDtypesPtrs)
            .OutputDataTypes(outputDtypesPtrs)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParam.group)},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.epWorldSize)},
                 {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.sendCounts)},
                 {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.recvCounts)},
                 {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transGmmWeight)},
                 {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transMmWeight)},
                 {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.yDtype)},
                 {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.mmDtype)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantGroupedMatMulAlltoAllv")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT8_E5M2);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT8_E5M2);
}

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferShapeTest_NormalWithMm)
{
    auto tilingParams = TilingParams{};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantGroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2},
              {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim0, tilingParams.mmWeightDim1},
              {tilingParams.mmWeightDim0, tilingParams.mmWeightDim1}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.sendCounts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.recvCounts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transGmmWeight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transMmWeight)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.yDtype)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.mmDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{tilingParams.BSK, tilingParams.N1},
                                                           {tilingParams.BS, tilingParams.N2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferShapeTest_NoMmInput)
{
    auto tilingParams = TilingParams{};

    // Optional mm_x/mm_weight use IR indices 6/7; omit them so optional shapes are nullptr (no MM branch).
    gert::InfershapeContextPara infershapeContextPara(
        "QuantGroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2},
              {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.sendCounts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.recvCounts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transGmmWeight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.yDtype)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.mmDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{tilingParams.BSK, tilingParams.N1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferShapeTest_TransGmmWeight)
{
    auto tilingParams = TilingParams{};
    tilingParams.transGmmWeight = true;

    gert::InfershapeContextPara infershapeContextPara(
        "QuantGroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim2, tilingParams.gmmWeightDim1},
              {tilingParams.e, tilingParams.gmmWeightDim2, tilingParams.gmmWeightDim1}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim0, tilingParams.mmWeightDim1},
              {tilingParams.mmWeightDim0, tilingParams.mmWeightDim1}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.sendCounts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.recvCounts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transGmmWeight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transMmWeight)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.yDtype)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.mmDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{tilingParams.BSK, tilingParams.gmmWeightDim2},
                                                           {tilingParams.BS, tilingParams.N2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantGroupedMatMulAlltoAllvInfershapeTest, InferShapeTest_TransMmWeight)
{
    auto tilingParams = TilingParams{};
    tilingParams.transMmWeight = true;

    gert::InfershapeContextPara infershapeContextPara(
        "QuantGroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2},
              {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.gmmWeightDim2}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim1, tilingParams.mmWeightDim0},
              {tilingParams.mmWeightDim1, tilingParams.mmWeightDim0}},
             ge::DT_HIFLOAT8,
             ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.sendCounts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.recvCounts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transGmmWeight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transMmWeight)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.yDtype)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.mmDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{tilingParams.BSK, tilingParams.N1},
                                                           {tilingParams.BS, tilingParams.mmWeightDim1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}
} // namespace QuantGroupedMatMulAlltoAllvUT
