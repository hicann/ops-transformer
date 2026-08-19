/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
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

constexpr const char *kOpName = "LightningIndexerGrad";

std::vector<OpAttr> MakeAttrs(const std::string &layout, int64_t sparseMode = 3, bool determinstic = false)
{
    return {
        {"headNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"determinstic", Ops::Transformer::AnyValue::CreateFrom<bool>(determinstic)}
    };
}

gert::InfershapeContextPara MakeBsndPara(ge::DataType dataType = ge::DT_FLOAT16,
                                         const std::string &layout = "BSND",
                                         std::initializer_list<int64_t> queryShape = {2, 8, 64, 128})
{
    return gert::InfershapeContextPara(
        kOpName,
        {
            {{queryShape, queryShape}, dataType, ge::FORMAT_ND},
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, dataType, ge::FORMAT_ND},
            {{{2, 8, 64, 128}, {2, 8, 64, 128}}, dataType, ge::FORMAT_ND},
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 8, 64}, {2, 8, 64}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, dataType, ge::FORMAT_ND}
        },
        MakeAttrs(layout));
}

gert::InfershapeContextPara MakeTndPara(ge::DataType dataType = ge::DT_BF16,
                                        const std::string &layout = "TND",
                                        std::initializer_list<int64_t> queryShape = {64, 64, 128})
{
    return gert::InfershapeContextPara(
        kOpName,
        {
            {{queryShape, queryShape}, dataType, ge::FORMAT_ND},
            {{{128, 1, 128}, {128, 1, 128}}, dataType, ge::FORMAT_ND},
            {{{64, 64, 128}, {64, 64, 128}}, dataType, ge::FORMAT_ND},
            {{{64, 2048}, {64, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, dataType, ge::FORMAT_ND}
        },
        MakeAttrs(layout));
}
} // namespace

class LightningIndexerGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerGradProto TearDown" << std::endl;
    }
};

TEST_F(LightningIndexerGradProto, infershape_bsnd_fp16_success)
{
    auto para = MakeBsndPara();
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 8, 64, 128},
        {2, 16, 1, 128},
        {2, 8, 64}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LightningIndexerGradProto, infershape_tnd_bf16_success)
{
    auto para = MakeTndPara();
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 64, 128},
        {128, 1, 128},
        {64, 64}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LightningIndexerGradProto, infershape_invalid_layout_failed)
{
    auto para = MakeBsndPara(ge::DT_FLOAT16, "BNSD");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradProto, infershape_bsnd_query_rank_failed)
{
    auto para = MakeBsndPara(ge::DT_FLOAT16, "BSND", {2, 8, 128});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradProto, infershape_tnd_query_rank_failed)
{
    auto para = MakeTndPara(ge::DT_BF16, "TND", {2, 8, 64, 128});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradProto, inferdtype_follow_query_dtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl(kOpName)->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType dataType = ge::DT_BF16;
    ge::DataType int32Type = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(7, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 0, 0})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&dataType, &dataType, &dataType, &int32Type, &dataType, &int32Type, &int32Type})
        .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dataType);
    EXPECT_EQ(context->GetOutputDataType(1), dataType);
    EXPECT_EQ(context->GetOutputDataType(2), dataType);
}
