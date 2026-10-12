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
 * \file test_QkvRmsNormRopeCache_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include <iostream>

using namespace std;

class QkvRmsNormRopeCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QkvRmsNormRopeCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QkvRmsNormRopeCache Proto Test TearDown" << std::endl;
    }
};

std::vector<int64_t> ToString(const gert::Shape& shape)
{
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);
    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeA)
{
    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    gert::InfershapeContextPara infershapeContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    std::vector<std::vector<int64_t>> expectOutputShape = {expectedoutput0, expectedoutput1, expectedoutput2,
                                                           expectedoutput3, expectedoutput4, expectedoutput5};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeB)
{
    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    gert::InfershapeContextPara infershapeContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    std::vector<std::vector<int64_t>> expectOutputShape = {expectedoutput0, expectedoutput1, expectedoutput2,
                                                           expectedoutput3, expectedoutput4, expectedoutput5};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeC)
{
    int batch_size = 16;
    int seq_len = 3;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    gert::InfershapeContextPara infershapeContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    std::vector<std::vector<int64_t>> expectOutputShape = {expectedoutput0, expectedoutput1, expectedoutput2,
                                                           expectedoutput3, expectedoutput4, expectedoutput5};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("QkvRmsNormRopeCache"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;

    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 128;

    if (data_type_func != nullptr) {
        ge::DataType input_0 = ge::DT_FLOAT16;
        ge::DataType input_1 = ge::DT_INT64;
        ge::DataType input_2 = ge::DT_FLOAT;
        ge::DataType input_3 = ge::DT_INT8;
        ge::DataType output_0 = ge::DT_FLOAT16;
        ge::DataType output_1 = ge::DT_INT8;

        string cache_mode("PA_NZ");
        vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
        vector<int64_t> head_nums = {Nq, Nk, Nv};

        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(13)
                .NodeIoNum(13, 6)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3,
                                 &input_3, &input_2, &input_2, &input_2, &input_2})
                .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_0);
        EXPECT_EQ(context->GetOutputDataType(1), output_1);
        EXPECT_EQ(context->GetOutputDataType(2), output_1);
        EXPECT_EQ(context->GetOutputDataType(3), output_0);
        EXPECT_EQ(context->GetOutputDataType(4), output_0);
        EXPECT_EQ(context->GetOutputDataType(5), output_0);
    }
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test2)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("QkvRmsNormRopeCache"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;

    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_INT8;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_INT8;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(13)
            .NodeIoNum(13, 6)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                        {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3, &input_3,
                             &input_2, &input_2, &input_2, &input_2})
            .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(5), ge::DT_FLOAT16);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test3)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("QkvRmsNormRopeCache"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;

    int batch_size = 16;
    int seq_len = 3;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_INT8;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_INT8;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(13)
            .NodeIoNum(13, 6)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                        {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3, &input_3,
                             &input_2, &input_2, &input_2, &input_2})
            .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(5), ge::DT_FLOAT16);
}
