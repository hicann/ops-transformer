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
#include "stem_indexer_param.h"
#include "infer_shape_case_executor.h"

namespace StemIndexerUT {

class StemIndexerInferShapeTest : public testing::TestWithParam<StemIndexerInferShapeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "StemIndexer InferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StemIndexer InferShapeTest TearDown" << std::endl; }
};

TEST_P(StemIndexerInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.qflat);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.kflat);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.vbias);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.q_seq_lens);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.kv_seq_lens);
    if (param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.num_prompt_tokens);
    if (param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.metadata);

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.sparse_indices);
    if (param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.sparse_seq_len);

    gert::InfershapeContextPara infershapeContextPara("StemIndexer", inputTensorDesc, outputTensorDesc);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    StemIndexer, StemIndexerInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<StemIndexerInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StemIndexerInferShapeUtParam>);

} // namespace StemIndexerUT
