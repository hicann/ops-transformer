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
#include "mc2_infer_shape_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulInferShapeTest : public testing::TestWithParam<AllGatherMatmulInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmul InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmul InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulInferShapeTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.bias);
    gert::InfershapeContextPara inferShapeContextPara(
        "AllGatherMatmul", inputTensorDesc, {param.y, param.gather_out},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)}},
        param.inputInstance, param.outputInstance);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    Mc2ExecuteTestCase(inferShapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmul, AllGatherMatmulInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulInferShapeUtParam>);

} // namespace AllGatherMatmulUT
