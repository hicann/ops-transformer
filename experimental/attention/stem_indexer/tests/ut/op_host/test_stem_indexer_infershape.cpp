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

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

class StemIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StemIndexerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StemIndexerProto TearDown" << std::endl;
    }
};

TEST_F(StemIndexerProto, StemIndexer_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StemIndexer",
        {
            {{{2, 32, 8, 2048}, {2, 32, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 16, 2048}, {2, 4, 16, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 16}, {2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 32, 8, 16},
        {2, 32, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StemIndexerProto, StemIndexer_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StemIndexer",
        {
            {{{1, 64, 4, 2048}, {1, 64, 4, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 32, 2048}, {1, 8, 32, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 32}, {1, 8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 4, 32},
        {1, 64, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StemIndexerProto, StemIndexer_infershape_qflat_dim_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StemIndexer",
        {
            {{{2, 32, 2048}, {2, 32, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 16, 2048}, {2, 4, 16, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 16}, {2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StemIndexerProto, StemIndexer_infershape_kflat_dim_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StemIndexer",
        {
            {{{2, 32, 8, 2048}, {2, 32, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 2048}, {2, 4, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 16}, {2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StemIndexerProto, StemIndexer_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("StemIndexer");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType qflatType = ge::DT_BF16;
    ge::DataType vbiasType = ge::DT_FLOAT;
    ge::DataType seqType = ge::DT_INT32;
    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(7, 2)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&qflatType, &qflatType, &vbiasType, &seqType, &seqType, &seqType, &seqType})
            .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}
