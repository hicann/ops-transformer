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
 * \file test_moeTokenUnpermute_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermute : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermute Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermute Proto Test TearDown" << std::endl;
  }
};

TEST_F(MoeTokenUnpermute, test_infershape_bf16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infershape_fp16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infershape_prob_none_bf16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infershape_prob_none_fp16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infershape_fp32) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infershape_prob_none_fp32) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{49152, 5120},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenUnpermute, test_infertype_bf16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("MoeTokenUnpermute"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

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
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_fp16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

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
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_fp32) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

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
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_bf16_fp32) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_bf16_fp16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_fp16_fp32) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_fp16_bf16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_fp32_bf16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(MoeTokenUnpermute, test_infertype_mix_fp32_fp16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeTokenUnpermute")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
