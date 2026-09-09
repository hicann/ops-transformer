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
 * \file test_quant_all_reduce_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace QuantAllReduceUT {

class QuantAllReduceInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantAllReduceInfershapeTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "QuantAllReduceInfershapeTest TearDown." << std::endl;
    }
};

// 2D x - K-G量化
TEST_F(QuantAllReduceInfershapeTest, InferShape2DKgInt8)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 5120}, {}};
    gert::StorageShape scalesShape = {{1024, 40}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D x - K-G量化 HIFLOAT8
TEST_F(QuantAllReduceInfershapeTest, InferShape2DKgHifloat8)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{2048, 7168}, {}};
    gert::StorageShape scalesShape = {{2048, 56}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D x - K-G量化 rank_size=4
TEST_F(QuantAllReduceInfershapeTest, InferShape2DKgRank4)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 4096}, {}};
    gert::StorageShape scalesShape = {{1024, 32}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_BF16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_BF16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D x - K-G量化 rank_size=2
TEST_F(QuantAllReduceInfershapeTest, InferShape2DKgRank2)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 4096}, {}};
    gert::StorageShape scalesShape = {{1024, 32}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D x - MX量化
TEST_F(QuantAllReduceInfershapeTest, InferShape2DMxE4m3)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 5120}, {}};
    gert::StorageShape scalesShape = {{1024, 80, 2}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D x - MX量化 E5M2
TEST_F(QuantAllReduceInfershapeTest, InferShape2DMxE5m2)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{2048, 7168}, {}};
    gert::StorageShape scalesShape = {{2048, 112, 2}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D x - K-G量化
TEST_F(QuantAllReduceInfershapeTest, InferShape3DKgInt8)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{8, 128, 5120}, {}};
    gert::StorageShape scalesShape = {{8, 128, 40}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 128, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D x - K-G量化 HIFLOAT8
TEST_F(QuantAllReduceInfershapeTest, InferShape3DKgHifloat8)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{16, 128, 7168}, {}};
    gert::StorageShape scalesShape = {{16, 128, 56}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 128, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D x - MX量化
TEST_F(QuantAllReduceInfershapeTest, InferShape3DMxE4m3)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{8, 128, 4096}, {}};
    gert::StorageShape scalesShape = {{8, 128, 64, 2}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 128, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D x - MX量化 E5M2
TEST_F(QuantAllReduceInfershapeTest, InferShape3DMxE5m2)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{8, 128, 5120}, {}};
    gert::StorageShape scalesShape = {{8, 128, 80, 2}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 128, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 异常 - 无效rank size
TEST_F(QuantAllReduceInfershapeTest, InferShapeInvalidRankSize)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 5120}, {}};
    gert::StorageShape scalesShape = {{1024, 40}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 3}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 异常 - x维度为1D（无效）
TEST_F(QuantAllReduceInfershapeTest, InferShapeInvalidDim1D)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{5120}, {}};
    gert::StorageShape scalesShape = {{40}, {}};
    gert::StorageShape outputShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantAllReduce",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// InferDataType - output_dtype 为 FLOAT16
TEST_F(QuantAllReduceInfershapeTest, InferDtypeFloat16)
{
    ge::DataType xType = ge::DT_INT8;
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 1)
            .NodeAttrs({{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
                        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                        {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
                        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, scalesType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &scalesType})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("QuantAllReduce");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// InferDataType - output_dtype 为 FLOAT
TEST_F(QuantAllReduceInfershapeTest, InferDtypeFloat)
{
    ge::DataType xType = ge::DT_HIFLOAT8;
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 1)
            .NodeAttrs({{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
                        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                        {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT)},
                        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, scalesType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &scalesType})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("QuantAllReduce");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
}

} // namespace QuantAllReduceUT
