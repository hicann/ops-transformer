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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PoolKeyIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PoolKeyIndexerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PoolKeyIndexerProto TearDown" << std::endl;
    }
};

// a) BSND degraded scenario (pool_size=1, maskMode=0, returnValue=true)
TEST_F(PoolKeyIndexerProto, PoolKeyIndexer_infershape_0)
{
    int64_t pool_tail_k_list[] = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "PoolKeyIndexer",
        // 输入Tensor
        {
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // pool_key
            {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weights
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, pool_tail_k_list},     // pool_tail_k
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_seq_q (null)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_seq_k (null)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // q_descale (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // k_descale (null)
        },
        {
            // 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {// 属性
         {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"pool_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 128, 128},
                                                           {1, 128, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// b) TND causal mask (pool_size=1, maskMode=3, returnValue=true)
TEST_F(PoolKeyIndexerProto, PoolKeyIndexer_infershape_1)
{
    int64_t actual_seq_qlist[] = {64, 64};
    int64_t actual_seq_kvlist[] = {128, 256};
    int64_t pool_tail_k_list[] = {0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "PoolKeyIndexer",
        // 输入Tensor
        {
            {{{128, 8, 128}, {128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // query (T1=128)
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pool_key (T2=256)
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // weights
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, pool_tail_k_list},  // pool_tail_k
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_k
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                          // block_table (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // q_descale (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // k_descale (null)
        },
        {
            // 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {// 属性
         {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"pool_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {128, 64},
                                                           {128, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// c) pool_size>1 (pool_size=4, maskMode=0, returnValue=true)
TEST_F(PoolKeyIndexerProto, PoolKeyIndexer_infershape_2)
{
    int64_t pool_tail_k_list[] = {2};
    gert::InfershapeContextPara infershapeContextPara(
        "PoolKeyIndexer",
        // 输入Tensor
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query
            {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // pool_key (S2=32 pool)
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weights
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, pool_tail_k_list},   // pool_tail_k
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                           // actual_seq_q (null)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                           // actual_seq_k (null)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                           // block_table (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // q_descale (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // k_descale (null)
        },
        {
            // 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {// 属性
         {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"pool_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 64, 131},
                                                           {1, 64, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// d) returnValue=false
TEST_F(PoolKeyIndexerProto, PoolKeyIndexer_infershape_3)
{
    int64_t pool_tail_k_list[] = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "PoolKeyIndexer",
        // 输入Tensor
        {
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // pool_key
            {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weights
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, pool_tail_k_list},     // pool_tail_k
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_seq_q (null)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_seq_k (null)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // q_descale (null)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // k_descale (null)
        },
        {
            // 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {// 属性
         {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"pool_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {// 预期输出
                                                           {1, 128, 128},
                                                           {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 比对成功返回SUCCESS
}

// infer dataType
TEST_F(PoolKeyIndexerProto, PoolKeyIndexer_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("PoolKeyIndexer")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref0 = ge::DT_FLOAT16; // query, pool_key, weights
        ge::DataType input_ref1 = ge::DT_INT64;   // pool_tail_k, actual_seq_q, actual_seq_k
        ge::DataType input_ref2 = ge::DT_FLOAT;   // q_descale, k_descale
        ge::DataType input_ref3 = ge::DT_INT32;   // block_table
        ge::DataType output_ref0 = ge::DT_INT32;  // sparse_indices
        ge::DataType output_ref1 = ge::DT_FLOAT;  // sparse_values
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(9, 2)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref0, &input_ref0, &input_ref0, &input_ref1, &input_ref1,
                                                   &input_ref1, &input_ref3, &input_ref2, &input_ref2})
                                  .NodeAttrs({{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                              {"pool_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref0);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref1);
    }
}
