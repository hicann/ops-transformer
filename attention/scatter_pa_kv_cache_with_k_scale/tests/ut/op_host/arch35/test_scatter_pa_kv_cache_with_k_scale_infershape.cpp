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
#include "scatter_pa_kv_cache_with_k_scale_param.h"
#include "infer_shape_case_executor.h"

namespace ScatterPaKvCacheWithKScaleUT {

class ScatterPaKvCacheWithKScaleInferShapeTest
    : public testing::TestWithParam<ScatterPaKvCacheWithKScaleInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterPaKvCacheWithKScale InferShapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ScatterPaKvCacheWithKScale InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(ScatterPaKvCacheWithKScaleInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.key);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.value);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.key_cache);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.value_cache);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.slot_mapping);
    if (param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.key_scale);
    if (param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.key_scale_cache);

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.key_cache_out);
    if (param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.value_cache_out);
    if (param.outputInstance[2] == 1)
        outputTensorDesc.emplace_back(param.key_scale_cache_out);

    gert::InfershapeContextPara infershapeContextPara(
        "ScatterPaKvCacheWithKScale", inputTensorDesc, outputTensorDesc,
        {{"cache_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.cache_layout)}});

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    ScatterPaKvCacheWithKScale, ScatterPaKvCacheWithKScaleInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<ScatterPaKvCacheWithKScaleInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScatterPaKvCacheWithKScaleInferShapeUtParam>);

} // namespace ScatterPaKvCacheWithKScaleUT
