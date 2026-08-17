/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using OpAttr = gert::InfershapeContextPara::OpAttr;

constexpr const char *kOpName = "LightningIndexerKLLoss";

std::vector<OpAttr> MakeAttrs(float eps = 1e-9f, bool deterministic = false)
{
    return {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
            {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(deterministic)}};
}

gert::InfershapeContextPara Make2dPara(ge::DataType dataType = ge::DT_FLOAT,
                                       std::initializer_list<int64_t> shape = {8, 22})
{
    return gert::InfershapeContextPara(
        kOpName, {{{shape, shape}, dataType, ge::FORMAT_ND}, {{shape, shape}, dataType, ge::FORMAT_ND}},
        {{{{1}, {1}}, dataType, ge::FORMAT_ND}}, MakeAttrs());
}

gert::InfershapeContextPara Make3dPara(ge::DataType dataType = ge::DT_FLOAT,
                                       std::initializer_list<int64_t> shape = {4, 10, 333})
{
    return gert::InfershapeContextPara(
        kOpName, {{{shape, shape}, dataType, ge::FORMAT_ND}, {{shape, shape}, dataType, ge::FORMAT_ND}},
        {{{{1}, {1}}, dataType, ge::FORMAT_ND}}, MakeAttrs());
}

gert::InfershapeContextPara Make1dPara(ge::DataType dataType = ge::DT_FLOAT, std::initializer_list<int64_t> shape = {8})
{
    return gert::InfershapeContextPara(
        kOpName, {{{shape, shape}, dataType, ge::FORMAT_ND}, {{shape, shape}, dataType, ge::FORMAT_ND}},
        {{{{1}, {1}}, dataType, ge::FORMAT_ND}}, MakeAttrs());
}

gert::InfershapeContextPara MakeShapeMismatchPara(ge::DataType dataType = ge::DT_FLOAT)
{
    return gert::InfershapeContextPara(kOpName,
                                       {{{{4, 10, 333}, {4, 10, 333}}, dataType, ge::FORMAT_ND},
                                        {{{4, 10, 111}, {4, 10, 111}}, dataType, ge::FORMAT_ND}},
                                       {{{{1}, {1}}, dataType, ge::FORMAT_ND}}, MakeAttrs());
}
} // namespace

class LightningIndexerKLLossProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LightningIndexerKLLossProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LightningIndexerKLLossProto TearDown" << std::endl; }
};

TEST_F(LightningIndexerKLLossProto, infershape_2d_fp32_success)
{
    auto para = Make2dPara();
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LightningIndexerKLLossProto, infershape_3d_fp16_success)
{
    auto para = Make3dPara(ge::DT_FLOAT16);
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LightningIndexerKLLossProto, infershape_3d_bf16_success)
{
    auto para = Make3dPara(ge::DT_BF16);
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LightningIndexerKLLossProto, inferdtype_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl(kOpName)->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType dataType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&dataType, &dataType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dataType);
}

TEST_F(LightningIndexerKLLossProto, inferdtype_bf16_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl(kOpName)->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType dataType = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&dataType, &dataType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dataType);
}
