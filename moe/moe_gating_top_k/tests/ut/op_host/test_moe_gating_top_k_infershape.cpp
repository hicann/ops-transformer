/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeGatingTopK : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopK SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopK TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopK",
        {
            {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 8}, {16, 8}, {16, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopK",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopK",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_03)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeGatingTopK",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"k_group", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"group_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"routed_scaling_factor", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_fail_01)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopK",
                                                      {
                                                          {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_fail_02)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopK",
                                                      {
                                                          {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_fail_03)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopK",
                                                      {
                                                          {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_fail_04)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopK",
                                                      {
                                                          {{{-3, 256}, {-3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_datatype_00)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType expertType = ge::DT_UNDEFINED;
    ge::DataType outType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType, &biasType})
                             .OutputDataTypes({&yType, &expertType, &outType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeGatingTopK")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_datatype_01)
{
    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType biasType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType expertType = ge::DT_UNDEFINED;
    ge::DataType outType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType, &biasType})
                             .OutputDataTypes({&yType, &expertType, &outType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeGatingTopK")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}
