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
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
constexpr int64_t CHUNK_SIZE = 64;

gert::InfershapeContextPara MakeContext(int64_t b, int64_t t, int64_t hk, int64_t hv, int64_t kDim, int64_t vDim)
{
    gert::StorageShape qShape = {{b, t, hk, kDim}, {b, t, hk, kDim}};
    gert::StorageShape vShape = {{b, t, hv, vDim}, {b, t, hv, vDim}};
    gert::StorageShape gShape = {{b, t, hv}, {b, t, hv}};
    return gert::InfershapeContextPara("ChunkGatedDeltaRuleComputeWy",
                                       {
                                           {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {vShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {gShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {"chunk_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(CHUNK_SIZE)},
                                       });
}
} // namespace

TEST(ChunkGatedDeltaRuleComputeWyInferShapeTest, TransposesToBhtdAndExpandsValueHeads)
{
    constexpr int64_t b = 2;
    constexpr int64_t t = 256;
    constexpr int64_t hk = 2;
    constexpr int64_t hv = 4;
    constexpr int64_t kDim = 64;
    constexpr int64_t vDim = 128;
    auto context = MakeContext(b, t, hk, hv, kDim, vDim);

    std::vector<std::vector<int64_t>> expected = {
        {b, hk, t, kDim}, // q_kernel
        {b, hk, t, kDim}, // k_kernel
        {b, hv, t, kDim}, // w_kernel
        {b, hv, t, vDim}, // u_kernel
        {b, hv, t},       // g_kernel
    };
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST(ChunkGatedDeltaRuleComputeWyInferShapeTest, RejectsWrongDimNum)
{
    gert::StorageShape qShape = {{1, 256, 2}, {1, 256, 2}};
    gert::StorageShape vShape = {{1, 256, 4, 64}, {1, 256, 4, 64}};
    gert::StorageShape gShape = {{1, 256, 4}, {1, 256, 4}};
    gert::InfershapeContextPara context("ChunkGatedDeltaRuleComputeWy",
                                        {
                                            {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {vShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {gShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {"chunk_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(CHUNK_SIZE)},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
