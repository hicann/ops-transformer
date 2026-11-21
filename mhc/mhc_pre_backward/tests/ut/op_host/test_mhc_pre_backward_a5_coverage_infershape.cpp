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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {

constexpr uint32_t N = 4;
constexpr uint32_t D = 64;
constexpr uint32_t FUSION_SIZE = N * N + 2 * N;

class MhcPreBackwardA5CoverageInferShapeTest : public testing::Test {};

TEST_F(MhcPreBackwardA5CoverageInferShapeTest, Tnd)
{
    constexpr uint32_t T = 8;
    constexpr uint32_t ND = N * D;
    gert::InfershapeContextPara context("MhcPreBackward",
                                        {
                                            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, FUSION_SIZE}, {T, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{T, N, D}, {FUSION_SIZE, ND}, {3}, {FUSION_SIZE}, {N, D}});
}

TEST_F(MhcPreBackwardA5CoverageInferShapeTest, Bsnd)
{
    constexpr uint32_t B = 2;
    constexpr uint32_t S = 4;
    constexpr uint32_t ND = N * D;
    gert::InfershapeContextPara context("MhcPreBackward",
                                        {
                                            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S, FUSION_SIZE}, {B, S, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{B, S, N, D}, {FUSION_SIZE, ND}, {3}, {FUSION_SIZE}, {N, D}});
}

TEST_F(MhcPreBackwardA5CoverageInferShapeTest, RejectInvalidResidualRank)
{
    constexpr uint32_t T = 8;
    constexpr uint32_t ND = N * D;
    gert::InfershapeContextPara context("MhcPreBackward",
                                        {
                                            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N, N, 1}, {T, N, N, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, FUSION_SIZE}, {T, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}

TEST_F(MhcPreBackwardA5CoverageInferShapeTest, InferAllOutputDataTypes)
{
    for (auto xType : {ge::DT_FLOAT16, ge::DT_BF16}) {
        ge::DataType floatType = ge::DT_FLOAT;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(12, 5)
                          .InputDataTypes({&xType, &floatType, &floatType, &xType, &floatType, &floatType, &floatType,
                                           &floatType, &floatType, &floatType, &floatType, &xType})
                          .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
        auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        auto infer = registry->GetOpImpl("MhcPreBackward")->infer_datatype;
        ASSERT_NE(infer, nullptr);
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(infer(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xType);
        for (size_t i = 1; i < 5; ++i)
            EXPECT_EQ(context->GetOutputDataType(i), ge::DT_FLOAT);
    }
}

} // namespace
