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
#include "quant_flash_attn_param.h"
#include "infer_shape_case_executor.h"

namespace QuantFlashAttnUT {

class QuantFlashAttnInferShapeTest : public testing::TestWithParam<QuantFlashAttnInferShapeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantFlashAttn InferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantFlashAttn InferShapeTest TearDown" << std::endl; }
};

// 临时添加，待框架修复后删除
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantFlashAttnInferShapeTest);

TEST_P(QuantFlashAttnInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.q);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.k);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.v);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.q_descale);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.k_descale);
    if (param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.v_descale);
    if (param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.block_table);
    if (param.inputInstance[7] == 1)
        inputTensorDesc.emplace_back(param.p_scale);
    if (param.inputInstance[8] == 1)
        inputTensorDesc.emplace_back(param.cu_seqlens_q);
    if (param.inputInstance[9] == 1)
        inputTensorDesc.emplace_back(param.cu_seqlens_kv);
    if (param.inputInstance[10] == 1)
        inputTensorDesc.emplace_back(param.seqused_q);
    if (param.inputInstance[11] == 1)
        inputTensorDesc.emplace_back(param.seqused_kv);
    if (param.inputInstance[12] == 1)
        inputTensorDesc.emplace_back(param.sinks);
    if (param.inputInstance[13] == 1)
        inputTensorDesc.emplace_back(param.attn_mask);
    if (param.inputInstance[14] == 1)
        inputTensorDesc.emplace_back(param.metadata);

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.attn_out);
    if (param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.softmax_lse);

    gert::InfershapeContextPara infershapeContextPara(
        "QuantFlashAttn", inputTensorDesc, outputTensorDesc,
        {{"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_compute_mode)},
         {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
         {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_left)},
         {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_right)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
         {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
         {"layout_q_descale", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q_descale)},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
         {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)}},
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    QuantFlashAttn, QuantFlashAttnInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantFlashAttnInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantFlashAttnInferShapeUtParam>);

} // namespace QuantFlashAttnUT
