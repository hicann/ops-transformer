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
#include "all_gather_matmul_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulInferDataTypeTest : public testing::TestWithParam<AllGatherMatmulInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmul InferDataTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmul InferDataTypeTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulInferDataTypeTest, param)
{
    auto param = GetParam();
    std::vector<void*> inputDataTypes;
    if (param.inputInstance[0] == 1)
        inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstance[1] == 1)
        inputDataTypes.emplace_back(&param.x2);
    if (param.inputInstance[2] == 1)
        inputDataTypes.emplace_back(&param.bias);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("AllGatherMatmul")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
                        {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
                        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
                        {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmul")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.y);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.gather_out);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmul, AllGatherMatmulInferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulInferDataTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulInferDataTypeUtParam>);

} // namespace AllGatherMatmulUT
