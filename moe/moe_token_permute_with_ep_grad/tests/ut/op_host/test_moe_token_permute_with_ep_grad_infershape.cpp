/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_datatype_context_faker.h" 
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenPermuteWithEpGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGrad Proto Test TearDown" << std::endl;
    }
};
TEST_F(MoeTokenPermuteWithEpGradInferShape, MoeTokenPermuteWithEpGrad_infershape_case_0)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    // output
    gert::StorageShape input_tokens_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{permuted_tokens_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
         {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
         {permuted_probs_output_d_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{input_tokens_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
         {input_probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermuteWithEpGradInferShape, infertype_bf16) {
  auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeTokenPermuteWithEpGrad")->infer_datatype;

  if (inferDtypeFunc != nullptr) {
    ge::DataType input_tokens_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType input_probs_ref = ge::DT_BF16;
    ge::DataType output_tokens_ref = ge::DT_BF16;
    ge::DataType output_probs_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 2)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_tokens_ref, &input_indices_ref, &input_probs_ref})
                              .OutputDataTypes({&output_tokens_ref, &output_probs_ref})
                              .Build();
    
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_tokens_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_tokens_ref);
  }
}