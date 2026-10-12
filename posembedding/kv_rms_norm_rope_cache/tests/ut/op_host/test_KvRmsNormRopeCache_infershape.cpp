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
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

using namespace std;

namespace {
const gert::InfershapeContextPara::TensorDescription kOptionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};

std::vector<gert::InfershapeContextPara::OpAttr> CommonAttrs()
{
    return {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    };
}
} // namespace

class KvRmsNormRopeCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test TearDown" << std::endl;
    }
};

// A path: kv shape [B,N,S,D] with full dims, no v input
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershapeA)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 512}, {64, 288, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 64}, {64, 288, 1, 512}, {64, 1, 1, 64}, {64, 1, 1, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// B path: kv shape [B,N,S,D] with smaller dims, has v input
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershapeB)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 192}, {64, 288, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 128}, {64, 288, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 192}, {64, 288, 1, 128}, {64, 1, 1, 192}, {64, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Dynamic shape A: kv_shape uses {-2}, output dims derive from cache shapes
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_2A)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 512}, {64, 288, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 64}, {64, 288, 1, 512}, {-1, -1, -1, 64}, {-1, -1, -1, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Dynamic shape B: kv_shape {-2}, v_shape also {-2}
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_2B)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 128}, {64, 288, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 64}, {64, 288, 1, 128}, {-1, -1, -1, 192}, {-1, -1, -1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Error dim A: 5D kv shape should cause GRAPH_FAILED
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_error_dimA)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 1, 576}, {64, 1, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 512}, {64, 288, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Error dim B: 5D kv_shape + 5D v_shape should cause GRAPH_FAILED
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_error_dimB)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 1, 192}, {64, 1, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 128}, {64, 288, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 1, 128}, {64, 1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Infer dtype test 1: all FLOAT16 inputs, FLOAT16 outputs
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test1)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("KvRmsNormRopeCache"),
              nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                              .GetSpaceRegistry()
                              ->GetOpImpl("KvRmsNormRopeCache")
                              ->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(12)
            .NodeIoNum(12, 4)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_0, &input_2, &input_2,
                             &input_2, &input_2})
            .OutputDataTypes({&output_0, &output_0, &output_0, &output_0})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

// Infer dtype test 2: all BF16 inputs, BF16 outputs
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test2)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("KvRmsNormRopeCache"),
              nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                              .GetSpaceRegistry()
                              ->GetOpImpl("KvRmsNormRopeCache")
                              ->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_BF16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_BF16;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(12)
            .NodeIoNum(12, 4)
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_0, &input_2, &input_2,
                             &input_2, &input_2, &input_0})
            .OutputDataTypes({&output_0, &output_0, &output_0, &output_0})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}

// Infer dtype test 3: FLOAT16 kv/gamma with INT8 cache, mixed output dtypes
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test3)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("KvRmsNormRopeCache"),
              nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                              .GetSpaceRegistry()
                              ->GetOpImpl("KvRmsNormRopeCache")
                              ->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_INT8;
    ge::DataType input_3 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_INT8;
    ge::DataType output_1 = ge::DT_FLOAT16;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(12)
            .NodeIoNum(12, 4)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_1, &input_2, &input_2, &input_3, &input_3,
                             &input_3, &input_3, &input_0})
            .OutputDataTypes({&output_0, &output_0, &output_1, &output_1})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

// Infer dtype test 4: BF16 kv/gamma with INT8 cache, mixed output dtypes
TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test4)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("KvRmsNormRopeCache"),
              nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                              .GetSpaceRegistry()
                              ->GetOpImpl("KvRmsNormRopeCache")
                              ->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_BF16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_INT8;
    ge::DataType input_3 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_INT8;
    ge::DataType output_1 = ge::DT_BF16;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(12)
            .NodeIoNum(12, 4)
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
                        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_1, &input_2, &input_2, &input_3, &input_3,
                             &input_3, &input_3, &input_0})
            .OutputDataTypes({&output_0, &output_0, &output_1, &output_1})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}
