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
#include "infershape_case_executor.h"

namespace {
std::vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(float clampMax = 4.6052F, float eps = 1.0e-12F)
{
    return {
        {"clamp_max", Ops::NN::AnyValue::CreateFrom<float>(clampMax)},
        {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
    };
}

void ExecuteDtypeCase(ge::DataType dtype)
{
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .SetOpType("ScaledCosineAttentionScore")
                      .Build();
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("ScaledCosineAttentionScore");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dtype);
}
} // namespace

TEST(ScaledCosineAttentionScoreInfershape, TypicalVideoMaeShape)
{
    gert::InfershapeContextPara para("ScaledCosineAttentionScore",
                                     {
                                         {{{1, 16, 2048, 88}, {1, 16, 2048, 88}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1, 16, 2048, 88}, {1, 16, 2048, 88}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{16, 1, 1}, {16, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, MakeAttrs());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 16, 2048, 2048}});
}

TEST(ScaledCosineAttentionScoreInfershape, OneDimensionalScale)
{
    gert::InfershapeContextPara para("ScaledCosineAttentionScore",
                                     {
                                         {{{2, 4, 17, 20}, {2, 4, 17, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4, 17, 20}, {2, 4, 17, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 17, 17}});
}

TEST(ScaledCosineAttentionScoreInfershape, DynamicSequence)
{
    gert::InfershapeContextPara para("ScaledCosineAttentionScore",
                                     {
                                         {{{1, 16, -1, 88}, {1, 16, -1, 88}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1, 16, -1, 88}, {1, 16, -1, 88}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}}, MakeAttrs());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 16, -1, -1}});
}

TEST(ScaledCosineAttentionScoreInfershape, QueryKeyMismatchFails)
{
    gert::InfershapeContextPara para("ScaledCosineAttentionScore",
                                     {
                                         {{{1, 4, 17, 20}, {1, 4, 17, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1, 4, 16, 20}, {1, 4, 16, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, MakeAttrs());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(ScaledCosineAttentionScoreInfershape, InvalidScaleFails)
{
    gert::InfershapeContextPara para("ScaledCosineAttentionScore",
                                     {
                                         {{{1, 4, 17, 20}, {1, 4, 17, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1, 4, 17, 20}, {1, 4, 17, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, MakeAttrs());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(ScaledCosineAttentionScoreInfershape, OutputDtypeFollowsQuery)
{
    ExecuteDtypeCase(ge::DT_FLOAT16);
    ExecuteDtypeCase(ge::DT_BF16);
    ExecuteDtypeCase(ge::DT_FLOAT);
}
