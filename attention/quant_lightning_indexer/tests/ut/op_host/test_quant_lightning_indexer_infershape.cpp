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

#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class QuantLightningIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerProto TearDown" << std::endl;
    }
};

TEST_F(QuantLightningIndexerProto, QuantLightningIndexer_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexer",
        {{{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 128, 1, 1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantLightningIndexerProto, QuantLightningIndexer_infershape_1)
{
    int64_t actualSeqQuery[] = {64, 128, 192};
    int64_t actualSeqKey[] = {256, 512, 768};
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexer",
        {{{{192, 16, 128}, {192, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{48, 16, 1, 128}, {48, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{192, 16}, {192, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 16}, {192, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{48, 16, 1}, {48, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
         {{{3, 48}, {3, 48}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{192, 1, 2048}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantLightningIndexerProto, QuantLightningIndexer_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("QuantLightningIndexer")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType queryType = ge::DT_INT8;
    ge::DataType scaleType = ge::DT_FLOAT16;
    ge::DataType seqType = ge::DT_INT32;
    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(8, 1)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&queryType, &queryType, &scaleType, &scaleType, &scaleType, &seqType, &seqType, &seqType})
            .NodeAttrs({{"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                        {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                        {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
                        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                        {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                        {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}})
            .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}
