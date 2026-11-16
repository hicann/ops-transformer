/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_hash_infershape.cpp
 * \brief EngramHash InferShape / InferDataType UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class EngramHashProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramHashProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramHashProto TearDown" << std::endl;
    }
};

// 模型默认形状 (N=3, L=2, tables=8): 输出应为 [L, T, C] = [2, 4096, 16]。
TEST_F(EngramHashProto, engram_hash_infershape_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "EngramHash",
        {
            {{{4096, 3}, {4096, 3}}, ge::DT_INT32, ge::FORMAT_ND}, // ngram_token_ids
            {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},       // multipliers
            {{{2, 2, 8}, {2, 2, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // vocab_sizes
            {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},     // offsets
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // hash_indices (输出 shape 应为 [2, 4096, 16])
        },
        {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// 空输入 (T=0): 输出应为 [L, 0, C]。
TEST_F(EngramHashProto, engram_hash_infershape_empty_tokens)
{
    gert::InfershapeContextPara infershapeContextPara(
        "EngramHash",
        {
            {{{0, 4}, {0, 4}}, ge::DT_INT32, ge::FORMAT_ND},       // ngram_token_ids
            {{{2, 4}, {2, 4}}, ge::DT_INT64, ge::FORMAT_ND},       // multipliers
            {{{2, 3, 8}, {2, 3, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // vocab_sizes
            {{{2, 24}, {2, 24}}, ge::DT_INT32, ge::FORMAT_ND},     // offsets
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // hash_indices (输出 shape 应为 [2, 0, 24])
        },
        {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(EngramHashProto, engram_hash_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("EngramHash")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputIdsDtype = ge::DT_INT32;
    ge::DataType inputMultDtype = ge::DT_INT64;
    ge::DataType inputVocabDtype = ge::DT_INT32;
    ge::DataType inputOffsDtype = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 1)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputIdsDtype, &inputMultDtype, &inputVocabDtype, &inputOffsDtype})
                              .OutputDataTypes({&outDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
}
