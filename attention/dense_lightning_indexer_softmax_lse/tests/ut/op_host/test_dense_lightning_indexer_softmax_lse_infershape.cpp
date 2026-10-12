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

class DenseLightningIndexerSoftmaxLseInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- DenseLightningIndexerSoftmaxLse InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- DenseLightningIndexerSoftmaxLse InferShape UT TearDown ---" << std::endl;
    }
};

TEST_F(DenseLightningIndexerSoftmaxLseInferShapeTest, infershape_bsnd_layout)
{
    int64_t b = 2, s1 = 256, s2 = 512, nidx1 = 16, nidx2 = 1, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "DenseLightningIndexerSoftmaxLse",
        {{{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, nidx2, s1}, {b, nidx2, s1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DenseLightningIndexerSoftmaxLseInferShapeTest, infershape_tnd_layout)
{
    int64_t b = 2, t1 = 512, t2 = 1024, nidx1 = 8, nidx2 = 1, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "DenseLightningIndexerSoftmaxLse",
        {{{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{nidx2, t1}, {nidx2, t1}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
