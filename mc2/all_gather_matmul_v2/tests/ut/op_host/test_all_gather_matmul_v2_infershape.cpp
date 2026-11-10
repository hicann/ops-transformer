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
#include "../../../../common/tests/ut/formal_error_assertions.h"
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

namespace {

// inferShape用例 ======================================================================================================
class AllGatherMatmulV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2InferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulV2InferShapeTest, Basic)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferShapeTest, EmptyTensorFailTest)
{
    gert::StorageShape x1Shape = {{8192, 0}, {}};
    gert::StorageShape x2Shape = {{0, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV2InferShapeTest, Pertensor)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};
    gert::StorageShape x1ScaleShape = {{1}, {}};
    gert::StorageShape x2ScaleShape = {{1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x1ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x2ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferShapeTest, Perblock)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};
    gert::StorageShape x1ScaleShape = {{64, 96}, {}};
    gert::StorageShape x2ScaleShape = {{96, 30}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x1ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x2ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// inferDtype用例 ======================================================================================================
class AllGatherMatmulV2InferDTypeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2InferDTypeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2InferDTypeTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulV2InferDTypeTest, YDtypeEqualX1DtypeFp16)
{
    ge::DataType x1Type = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(1, 2)
                             .InputDataTypes({&x1Type})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("AllGatherMatmulV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1Type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, YDtypeEqualX1DtypeBf16)
{
    ge::DataType x1Type = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(1, 2)
                             .InputDataTypes({&x1Type})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("AllGatherMatmulV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1Type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, AttrYDtype)
{
    ge::DataType x1Type = ge::DT_FLOAT8_E5M2;
    ge::DataType attrYDtype = ge::DT_FLOAT;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(1, 2)
            .InputDataTypes({&x1Type})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                        {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(attrYDtype))}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), attrYDtype);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1Type);
}

TEST_F(AllGatherMatmulV2InferShapeTest, AmaxOutTrue)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}, {0}, {1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferShapeTest, AmaxOutTrueFP8)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};
    gert::StorageShape x1ScaleShape = {{1}, {}};
    gert::StorageShape x2ScaleShape = {{1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x1ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x2ScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}, {65536, 12288}, {1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, FP8E4M3FN)
{
    ge::DataType x1Type = ge::DT_FLOAT8_E4M3FN;
    ge::DataType attrYDtype = ge::DT_FLOAT16;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(1, 2)
            .InputDataTypes({&x1Type})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                        {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(attrYDtype))}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), attrYDtype);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1Type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, HIFLOAT8)
{
    ge::DataType x1Type = ge::DT_HIFLOAT8;
    ge::DataType attrYDtype = ge::DT_BF16;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(1, 2)
            .InputDataTypes({&x1Type})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                        {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(attrYDtype))}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), attrYDtype);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1Type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, UndefinedYDtypeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    ge::DataType x1Type = ge::DT_INT8;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(1, 2)
            .InputDataTypes({&x1Type})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                 {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                 {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_UNDEFINED))}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
    ExpectExcelFormalError("EZ0026", "y_dtype", "DT_UNDEFINED");
}

TEST_F(AllGatherMatmulV2InferDTypeTest, MissingYDtypeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    ge::DataType x1Type = ge::DT_INT8;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(1, 2)
                             .InputDataTypes({&x1Type})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                                         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
    ExpectExcelFormalError("EZ0026", "y_dtype", "null");
}

TEST_F(AllGatherMatmulV2InferDTypeTest, InputTypeAndYDtypeStateContractMatrix)
{
    for (ge::DataType input : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}) {
        for (int state : {0, 1, 2, 3, 4}) {
            SCOPED_TRACE(std::to_string(input) + ":" + std::to_string(state));
            ExcelErrorReportScope errors;
            std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> attrs = {
                {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}};
            const ge::DataType outputs[] = {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_FLOAT16, ge::DT_BF16,
                                            ge::DT_FLOAT};
            if (state != 0)
                attrs.emplace_back("y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputs[state]));
            auto holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(1, 2)
                              .InputDataTypes({&input})
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(attrs)
                              .Build();
            auto* context = holder.GetContext<gert::InferDataTypeContext>();
            ASSERT_NE(context, nullptr);
            auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
            const bool invalid = input == ge::DT_INT8 && state < 2;
            EXPECT_EQ(registry->GetOpImpl("AllGatherMatmulV2")->infer_datatype(context),
                      invalid ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS);
            if (invalid) {
                ExpectExcelFormalError("EZ0026", "y_dtype", state == 0 ? "null" : "DT_UNDEFINED",
                                       "Set y_dtype to a defined data type when the input is not FLOAT16 or BF16");
            } else {
                EXPECT_EQ(context->GetOutputDataType(0), input == ge::DT_INT8 ? outputs[state] : input);
                EXPECT_EQ(context->GetOutputDataType(1), input);
                EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
            }
        }
    }
}
} // namespace
