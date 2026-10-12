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
#include <iostream>
#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using OpAttr = gert::InfershapeContextPara::OpAttr;

constexpr const char* kOpName = "SparseLightningIndexerGradKLLoss";

std::vector<OpAttr> MakeAttrs(const std::string& layout)
{
    return {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}};
}

gert::InfershapeContextPara MakeBsndPara(ge::DataType dataType, ge::DataType weightType)
{
    constexpr int64_t b = 2;
    constexpr int64_t s1 = 128;
    constexpr int64_t s2 = 256;
    constexpr int64_t n1 = 32;
    constexpr int64_t n2 = 1;
    constexpr int64_t nIdx = 8;
    constexpr int64_t d = 512;
    constexpr int64_t dIdx = 128;
    constexpr int64_t dRope = 64;
    constexpr int64_t topK = 2048;

    return gert::InfershapeContextPara(kOpName,
                                       {{{{b, s1, n1, d}, {b, s1, n1, d}}, dataType, ge::FORMAT_ND},
                                        {{{b, s2, n2, d}, {b, s2, n2, d}}, dataType, ge::FORMAT_ND},
                                        {{{b, s1, nIdx, dIdx}, {b, s1, nIdx, dIdx}}, dataType, ge::FORMAT_ND},
                                        {{{b, s2, n2, dIdx}, {b, s2, n2, dIdx}}, dataType, ge::FORMAT_ND},
                                        {{{b, s1, nIdx}, {b, s1, nIdx}}, weightType, ge::FORMAT_ND},
                                        {{{b, n2, s1, topK}, {b, n2, s1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{b, n2, s1, nIdx}, {b, n2, s1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{b, n2, s1, nIdx}, {b, n2, s1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{b, s1, n1, dRope}, {b, s1, n1, dRope}}, dataType, ge::FORMAT_ND},
                                        {{{b, s2, n2, dRope}, {b, s2, n2, dRope}}, dataType, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                       {{{{}, {}}, dataType, ge::FORMAT_ND},
                                        {{{}, {}}, dataType, ge::FORMAT_ND},
                                        {{{}, {}}, weightType, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                       MakeAttrs("BSND"));
}

gert::InfershapeContextPara MakeTndPara()
{
    constexpr int64_t t1 = 128;
    constexpr int64_t t2 = 256;
    constexpr int64_t n1 = 64;
    constexpr int64_t n2 = 1;
    constexpr int64_t nIdx = 16;
    constexpr int64_t d = 512;
    constexpr int64_t dIdx = 128;
    constexpr int64_t dRope = 64;
    constexpr int64_t topK = 2048;

    return gert::InfershapeContextPara(kOpName,
                                       {{{{t1, n1, d}, {t1, n1, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{t2, n2, d}, {t2, n2, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{t1, nIdx, dIdx}, {t1, nIdx, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{t2, n2, dIdx}, {t2, n2, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{t1, nIdx}, {t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{n2, t1, topK}, {n2, t1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{n2, t1, nIdx}, {n2, t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{n2, t1, nIdx}, {n2, t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{t1, n1, dRope}, {t1, n1, dRope}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{t2, n2, dRope}, {t2, n2, dRope}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                       {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                       MakeAttrs("TND"));
}
} // namespace

class SparseLightningIndexerGradKLLossProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerGradKLLossProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerGradKLLossProto TearDown" << std::endl;
    }
};

TEST_F(SparseLightningIndexerGradKLLossProto, infershape_bsnd_fp16)
{
    auto para = MakeBsndPara(ge::DT_FLOAT16, ge::DT_FLOAT16);
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 128, 8, 128}, {2, 256, 1, 128}, {2, 128, 8}, {1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerGradKLLossProto, infershape_bsnd_bf16_float_weight)
{
    auto para = MakeBsndPara(ge::DT_BF16, ge::DT_FLOAT);
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 128, 8, 128}, {2, 256, 1, 128}, {2, 128, 8}, {1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerGradKLLossProto, infershape_tnd_bf16)
{
    auto para = MakeTndPara();
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 16, 128}, {256, 1, 128}, {128, 16}, {1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerGradKLLossProto, inferdtype_fp16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl(kOpName)->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType int32Type = ge::DT_INT32;
    ge::DataType floatType = ge::DT_FLOAT;
    ge::DataType int64Type = ge::DT_INT64;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(12, 4)
                             .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&dataType, &dataType, &dataType, &dataType, &weightType, &int32Type,
                                              &floatType, &floatType, &dataType, &dataType, &int64Type, &int64Type})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dataType);
    EXPECT_EQ(context->GetOutputDataType(1), dataType);
    EXPECT_EQ(context->GetOutputDataType(2), weightType);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT);
}
