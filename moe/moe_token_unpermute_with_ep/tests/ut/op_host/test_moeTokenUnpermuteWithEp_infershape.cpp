/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moeTokenUnpermuteWithEp_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermuteWithEp : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEp Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEp Proto Test TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_bf16)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape probs_shape = {{6144, 8}, {6144, 8}};
    std::vector<int64_t> restore_shape({});
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        {probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_prob_none_bf16)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};
    std::vector<int64_t> restore_shape({});

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        //{probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_fp16)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape probs_shape = {{6144, 8}, {6144, 8}};
    std::vector<int64_t> restore_shape({});
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        {probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_prob_none_fp16)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};
    std::vector<int64_t> restore_shape({});

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        //{probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_fp32)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape probs_shape = {{6144, 8}, {6144, 8}};
    std::vector<int64_t> restore_shape({});
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        {probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infershape_prob_none_fp32)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_shape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    // output
    gert::StorageShape unpermuted_tokens_shape = {{6144, 5120}, {6144, 5120}};
    std::vector<int64_t> restore_shape({});

    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteWithEp",
    { // input info
        {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
        {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
        //{probs_shape, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    }, 
    { // attr
        {"num_topk",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermuteWithEp, test_infertype_bf16)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteWithEp, test_infertype_fp16)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteWithEp, test_infertype_fp32)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEp")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}