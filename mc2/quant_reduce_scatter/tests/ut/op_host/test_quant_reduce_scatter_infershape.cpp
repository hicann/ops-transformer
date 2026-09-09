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
 * \file test_quant_reduce_scatter_infershape.cpp
 * \brief infershape ut
 */

#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"

class QuantReduceScatterInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantReduceScatterInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantReduceScatterInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(QuantReduceScatterInferShapeTest, Basic2D_Rank8)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{1024, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{1024, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Basic2D_Rank2)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{1024, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{1024, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Basic2D_Rank4)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{1024, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{1024, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Basic3D_Rank8)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{8, 128, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{8, 128, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Basic3D_Rank2)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{4, 256, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{4, 256, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Dynamic3D)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{-1, 128, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{-1, 128, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Dynamic2D)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{-1, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{-1, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 5120}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, Dim1D)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{5120}, {}};
    gert::StorageShape scalesStorageShape = {{40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 0}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantReduceScatterInferShapeTest, InvalidRankSize)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{1024, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{1024, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 3}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(QuantReduceScatterInferShapeTest, Dim4D)
{
    gert::StorageShape contextStorageShape = {{1}, {}};
    gert::StorageShape xStorageShape = {{2, 8, 128, 5120}, {}};
    gert::StorageShape scalesStorageShape = {{2, 8, 128, 40}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "QuantReduceScatter",
        {{contextStorageShape, ge::DT_INT32, ge::FORMAT_ND},
         {xStorageShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(314572800)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 0}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}
