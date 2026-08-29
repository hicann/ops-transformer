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
#include "test_allto_all_matmul_v2_host_ut_param.h"
#include "infer_shape_case_executor.h"

namespace AlltoAllMatmulV2UT {

class InferShapeTest : public testing::TestWithParam<AlltoAllMatmulV2InferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulV2 InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulV2 InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(InferShapeTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    // context: op_def 第 0 个输入，InferShape 不使用但需占位以保持索引一致
    inputTensorDesc.emplace_back(param.context);
    if (param.inputInstance[1] == 1) {
        inputTensorDesc.emplace_back(param.x1);
    }
    if (param.inputInstance[2] == 1) {
        inputTensorDesc.emplace_back(param.x2);
    }
    if (param.inputInstance[3] == 1) {
        inputTensorDesc.emplace_back(param.bias);
    }
    if (param.inputInstance[4] == 1) {
        inputTensorDesc.emplace_back(param.x1Scale);
    }
    if (param.inputInstance[5] == 1) {
        inputTensorDesc.emplace_back(param.x2Scale);
    }

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc({param.y, param.all2allOut});

    gert::InfershapeContextPara inferShapeContextPara(
        "AlltoAllMatmulV2", inputTensorDesc, outputTensorDesc,
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.world_size)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype_attr)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transpose_x2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("urma")},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(inferShapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmulV2, InferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulV2InferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulV2InferShapeUtParam>);

} // namespace AlltoAllMatmulV2UT
