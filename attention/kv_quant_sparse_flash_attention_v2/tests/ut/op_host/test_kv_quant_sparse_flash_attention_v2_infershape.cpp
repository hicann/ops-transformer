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

class KvQuantSparseFlashAttentionV2Proto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionV2Proto TearDown" << std::endl;
    }
};

TEST_F(KvQuantSparseFlashAttentionV2Proto, KvQuantSparseFlashAttentionV2_infershape_0)
{
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "KvQuantSparseFlashAttentionV2",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});

    // kvHeadNum = key.dim[2] = 1, g = query.dim[2] / kvHeadNum = 8
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 8, 512}, {2, 1, 64, 8}, {2, 1, 64, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KvQuantSparseFlashAttentionV2Proto, KvQuantSparseFlashAttentionV2_infershape_1)
{
    int32_t actualSeqQuery[] = {48, 96};
    int32_t actualSeqKv[] = {128, 256};
    gert::InfershapeContextPara infershapeContextPara(
        "KvQuantSparseFlashAttentionV2",
        {{{{96, 16, 576}, {96, 16, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{96, 1, 64}, {96, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    // return_softmax_lse=false时softmax_max/softmax_sum输出shape为{0}
    std::vector<std::vector<int64_t>> expectOutputShape = {{96, 16, 512}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KvQuantSparseFlashAttentionV2Proto, KvQuantSparseFlashAttentionV2_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("KvQuantSparseFlashAttentionV2")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType queryType = ge::DT_BF16;
    ge::DataType kvType = ge::DT_INT8;
    ge::DataType indexType = ge::DT_INT32;
    ge::DataType scaleType = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(9, 1)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&queryType, &kvType, &kvType, &indexType, &scaleType, &scaleType,
                                              &indexType, &indexType, &indexType})
                             .NodeAttrs({{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
                                         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                                         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
