/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

class DenseLightningIndexerGradKLLossInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- DenseLightningIndexerGradKLLoss InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- DenseLightningIndexerGradKLLoss InferShape UT TearDown ---" << std::endl;
    }
};

TEST_F(DenseLightningIndexerGradKLLossInferShapeTest, infershape_bsnd_layout)
{
    int64_t b = 2, s1 = 256, s2 = 512, n1 = 64, n2 = 64, nidx1 = 16, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;

    gert::InfershapeContextPara infershapeContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{b, s1, n1, d}, {b, s1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, d}, {b, s2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, s1, n1, dr}, {b, s1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, dr}, {b, s2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, s1, nidx1, d}, {b, s2, nidx2, d}, {b, s1, nidx1}, {1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DenseLightningIndexerGradKLLossInferShapeTest, infershape_tnd_layout)
{
    int64_t t1 = 512, t2 = 1024, n1 = 32, n2 = 32, nidx1 = 8, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    int64_t batch = 2;

    gert::InfershapeContextPara infershapeContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{t1, n1, d}, {t1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, d}, {t2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{t1, n1, dr}, {t1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, dr}, {t2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{t1, nidx1, d}, {t2, nidx2, d}, {t1, nidx1}, {1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
