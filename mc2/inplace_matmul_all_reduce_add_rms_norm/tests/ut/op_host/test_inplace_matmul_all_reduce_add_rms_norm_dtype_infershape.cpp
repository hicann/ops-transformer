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
#include "inplace_matmul_all_reduce_add_rms_norm_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace InplaceMatmulAllReduceAddRmsNormUT {

class InferDataTypeInplaceMatmulAllReduceAddRmsNormTest
    : public testing::TestWithParam<InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNorm InferDataTypeInplaceMatmulAllReduceAddRmsNormTest SetUp"
                  << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNorm InferDataTypeInplaceMatmulAllReduceAddRmsNormTest TearDown"
                  << std::endl;
    }
};

TEST_P(InferDataTypeInplaceMatmulAllReduceAddRmsNormTest, param)
{
    auto param = GetParam();
    std::vector<void*> inputDataTypes;
    if (param.inputInstance.size() < 8) {
        return;
    }
    if (param.inputInstance[0] == 1)
        inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstance[1] == 1)
        inputDataTypes.emplace_back(&param.x2);
    if (param.inputInstance[2] == 1)
        inputDataTypes.emplace_back(&param.bias);
    if (param.inputInstance[3] == 1)
        inputDataTypes.emplace_back(&param.residual);
    if (param.inputInstance[4] == 1)
        inputDataTypes.emplace_back(&param.gamma);
    if (param.inputInstance[5] == 1)
        inputDataTypes.emplace_back(&param.antiquant_scale);
    if (param.inputInstance[6] == 1)
        inputDataTypes.emplace_back(&param.antiquant_offset);
    if (param.inputInstance[7] == 1)
        inputDataTypes.emplace_back(&param.dequant_scale);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("InplaceMatmulAllReduceAddRmsNorm")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
                 {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
                 {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
                 {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
                 {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
                 {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
                 {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(param.epsilon)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("InplaceMatmulAllReduceAddRmsNorm");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.expect_y_dtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
                  param.expect_norm_dtype);
    }
}

INSTANTIATE_TEST_SUITE_P(InplaceMatmulAllReduceAddRmsNorm, InferDataTypeInplaceMatmulAllReduceAddRmsNormTest,
                         testing::ValuesIn(GetCasesFromCsv<InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam>(
                             ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam>);

} // namespace InplaceMatmulAllReduceAddRmsNormUT
