/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rope_quant_kvcache_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RopeQuantKvcache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeQuantKvcache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeQuantKvcache TearDown" << std::endl;
    }
};

TEST_F(RopeQuantKvcache, RopeQuantKvcache_infershape_case_0)
{
    std::vector<int64_t> size_splits = {1024, 128, 128};
    gert::InfershapeContextPara infershapeContextPara(
        "RopeQuantKvcache",
        {
            // input info
            {{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            // attr
            {"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(size_splits)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 8, 128}, {4, 1, 1, 128}, {4, 1, 1, 128}, {4, 2048, 1, 128}, {4, 2048, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RopeQuantKvcache, RopeQuantKvcache_infertype_fp16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("RopeQuantKvcache"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("RopeQuantKvcache")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType qkv_ref = ge::DT_FLOAT16;
        ge::DataType cos_ref = ge::DT_FLOAT16;
        ge::DataType sin_ref = ge::DT_FLOAT16;
        ge::DataType qscale_ref = ge::DT_FLOAT;
        ge::DataType qoffset_ref = ge::DT_INT32;
        ge::DataType kcache_ref = ge::DT_INT8;
        ge::DataType vcache_ref = ge::DT_INT8;
        ge::DataType indice_ref = ge::DT_INT32;
        ge::DataType output_q_ref = ge::DT_FLOAT16;
        ge::DataType output_k_ref = ge::DT_FLOAT16;
        ge::DataType output_v_ref = ge::DT_FLOAT16;
        ge::DataType output_kcache_ref = ge::DT_INT8;
        ge::DataType output_vcache_ref = ge::DT_INT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(8)
                .NodeIoNum(8, 5)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes(
                    {&qkv_ref, &cos_ref, &sin_ref, &qscale_ref, &qoffset_ref, &kcache_ref, &vcache_ref, &indice_ref})
                .OutputDataTypes({&output_q_ref, &output_k_ref, &output_v_ref, &output_kcache_ref, &output_vcache_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), qkv_ref);
        EXPECT_EQ(context->GetInputDataType(1), cos_ref);
        EXPECT_EQ(context->GetInputDataType(2), sin_ref);
        EXPECT_EQ(context->GetInputDataType(3), qscale_ref);
        EXPECT_EQ(context->GetInputDataType(4), qoffset_ref);
        EXPECT_EQ(context->GetInputDataType(5), kcache_ref);
        EXPECT_EQ(context->GetInputDataType(6), vcache_ref);
        EXPECT_EQ(context->GetInputDataType(7), indice_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_q_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_k_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_v_ref);
        EXPECT_EQ(context->GetOutputDataType(3), output_kcache_ref);
        EXPECT_EQ(context->GetOutputDataType(4), output_vcache_ref);
    }
}
