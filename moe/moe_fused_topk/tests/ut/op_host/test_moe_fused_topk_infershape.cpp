/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_fused_topk_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeFusedTopk : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFusedTopk SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFusedTopk TearDown" << std::endl;
    }
};

static std::vector<gert::InfershapeContextPara::OpAttr> MakeFusedTopkAttrs(const int64_t topK)
{
    return {
        {"group_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"group_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"top_n", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topK)},
        {"activate_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"is_norm", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"enable_expert_mapping", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 8}, {16, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_01)
{
    // 部分未知维：x[0] = -1，输出行数保持 -1，最后一维由 topK 决定
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-1, 64}, {-1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_02)
{
    // 未知秩：x = (-2,)，输出行数保持未知 -1，最后一维由 topK 决定
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_01)
{
    // x 为 1D 且非 -2，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_02)
{
    // x 为 3D，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_03)
{
    // topK 为 0，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(0));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_04)
{
    // x 维度小于 -1，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-3, 64}, {-3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_05)
{
    // expertNum 维度小于 -1，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, -3}, {16, -3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_03)
{
    // 全未知维：x = [-1, -1]，expertNum 未知时不比较 add_num 长度与 topK 上限
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_04)
{
    // add_num 未知秩(-2)：秩与长度校验均豁免，x 已知时输出保持已知
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 8}, {16, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_05)
{
    // x 未知秩且 add_num 已知 1D：秩校验通过，expertNum 未知时仅跳过长度相等比较
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}, {-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_06)
{
    // topK 大于 expertNum，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 4}, {16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_07)
{
    // add_num 为 2D，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_08)
{
    // add_num 长度与 expertNum 不一致，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{32}, {32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_shape_fail_09)
{
    // x 未知秩但 add_num 已知为 2D，非法
    gert::InfershapeContextPara infershapeContextPara("MoeFusedTopk",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeFusedTopkAttrs(8));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_datatype_00)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType addNumType = ge::DT_FLOAT16;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType indicesType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 2)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType, &addNumType})
                             .OutputDataTypes({&yType, &indicesType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeFusedTopk")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(MoeFusedTopk, moe_fused_topk_infer_datatype_01)
{
    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType addNumType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType indicesType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 2)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xType, &addNumType})
                             .OutputDataTypes({&yType, &indicesType})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeFusedTopk")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}
