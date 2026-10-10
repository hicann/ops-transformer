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
#include "matmul_all_reduce_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MatmulAllReduceUT {

class InferDataTypeMatmulAllReduceTest : public testing::TestWithParam<MatmulAllReduceInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce InferDataTypeMatmulAllReduceTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce InferDataTypeMatmulAllReduceTest TearDown" << std::endl;
    }
};

TEST_P(InferDataTypeMatmulAllReduceTest, param)
{
    auto param = GetParam();
    std::vector<void*> inputDataTypes;
    if (param.inputInstance[0] == 1)
        inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstance[1] == 1)
        inputDataTypes.emplace_back(&param.x2);
    if (param.inputInstance[2] == 1)
        inputDataTypes.emplace_back(&param.bias);
    if (param.inputInstance[3] == 1)
        inputDataTypes.emplace_back(&param.x3);
    if (param.inputInstance[4] == 1)
        inputDataTypes.emplace_back(&param.antiquant_scale);
    if (param.inputInstance[5] == 1)
        inputDataTypes.emplace_back(&param.antiquant_offset);
    if (param.inputInstance[6] == 1)
        inputDataTypes.emplace_back(&param.dequant_scale);
    if (param.inputInstance[7] == 1)
        inputDataTypes.emplace_back(&param.pertoken_scale);
    if (param.inputInstance[8] == 1)
        inputDataTypes.emplace_back(&param.comm_quant_scale_1);
    if (param.inputInstance[9] == 1)
        inputDataTypes.emplace_back(&param.comm_quant_scale_2);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("MatmulAllReduce")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
                 {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
                 {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
                 {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
                 {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
                 {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MatmulAllReduce")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.y);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, InferDataTypeMatmulAllReduceTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceInferDataTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceInferDataTypeUtParam>);

} // namespace MatmulAllReduceUT
