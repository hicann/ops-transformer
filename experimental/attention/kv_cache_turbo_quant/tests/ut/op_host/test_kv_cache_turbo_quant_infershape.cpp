/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include <gtest/gtest.h>

#include "infer_shape_case_executor.h"

namespace {
using Attr = gert::InfershapeContextPara::OpAttr;

std::vector<Attr> MakeAttrs(int64_t bits, int64_t outputDtype)
{
    return {
        {"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bits)},
        {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtype)},
    };
}

void RunShapeCase(int64_t tokens, int64_t heads, int64_t headDim, int64_t qjlDim, int64_t bits, int64_t outputDtype)
{
    gert::InfershapeContextPara para(
        "KvCacheTurboQuant",
        {
            {{{tokens, heads, headDim}, {tokens, heads, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim, headDim}, {headDim, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{qjlDim, headDim}, {qjlDim, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, outputDtype == 1 ? ge::DT_FLOAT16 : ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, outputDtype == 1 ? ge::DT_FLOAT16 : ge::DT_BF16, ge::FORMAT_ND},
        },
        MakeAttrs(bits, outputDtype));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS,
                    {{tokens, heads, (headDim * bits + 7) / 8},
                     {tokens, heads, (qjlDim + 7) / 8},
                     {tokens, heads},
                     {tokens, heads}});
}
} // namespace

TEST(KvCacheTurboQuantInfershape, TwoBitGenericDimensions)
{
    RunShapeCase(5, 4, 64, 256, 2, 0);
}

TEST(KvCacheTurboQuantInfershape, ThreeBitFixedDimensions)
{
    RunShapeCase(1, 8, 128, 128, 3, 0);
}

TEST(KvCacheTurboQuantInfershape, FourBitFp16Metadata)
{
    RunShapeCase(2, 32, 256, 64, 4, 1);
}

TEST(KvCacheTurboQuantInfershape, RejectsRankTwoKvVectors)
{
    gert::InfershapeContextPara para("KvCacheTurboQuant",
                                     {
                                         {{{5, 64}, {5, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{256, 64}, {256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(2, 0));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(KvCacheTurboQuantInfershape, RejectsScalarQjlMatrix)
{
    gert::InfershapeContextPara para("KvCacheTurboQuant",
                                     {
                                         {{{5, 4, 64}, {5, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(2, 0));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(KvCacheTurboQuantInfershape, RejectsUnsupportedMseBits)
{
    gert::InfershapeContextPara para("KvCacheTurboQuant",
                                     {
                                         {{{5, 4, 64}, {5, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{256, 64}, {256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(5, 0));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(KvCacheTurboQuantInfershape, RejectsOverflowingMseBits)
{
    gert::InfershapeContextPara para("KvCacheTurboQuant",
                                     {
                                         {{{5, 4, 64}, {5, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{256, 64}, {256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(std::numeric_limits<int64_t>::max(), 0));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
