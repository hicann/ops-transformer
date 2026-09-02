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
#include "flash_attention_score_param.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace FlashAttentionScoreUT {

class FlashAttentionScoreInferShapeTest : public testing::TestWithParam<FlashAttentionScoreInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScore InferShapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScore InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(FlashAttentionScoreInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensors;
    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensors;

    gert::InfershapeContextPara::TensorDescription allInputs[] = {
        param.query,        param.key,          param.value,   param.real_shift,      param.drop_mask,
        param.padding_mask, param.atten_mask,   param.prefix,  param.actual_seq_qlen, param.actual_seq_kvlen,
        param.q_start_idx,  param.kv_start_idx, param.dScaleQ, param.dScaleK,         param.dScaleV,
        param.queryRope,    param.keyRope};
    for (size_t i = 0; i < param.inputInstance.size() && i < sizeof(allInputs) / sizeof(allInputs[0]); i++) {
        if (param.inputInstance[i] == 1) {
            inputTensors.emplace_back(allInputs[i]);
        }
    }

    gert::InfershapeContextPara::TensorDescription allOutputs[] = {param.softmaxMax, param.softmaxSum, param.softmaxOut,
                                                                   param.attentionOut};
    for (size_t i = 0; i < param.outputInstance.size() && i < sizeof(allOutputs) / sizeof(allOutputs[0]); i++) {
        if (param.outputInstance[i] == 1) {
            outputTensors.emplace_back(allOutputs[i]);
        }
    }

    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore", inputTensors, outputTensors,
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale_value)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(param.keep_prob)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tockens)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tockens)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_num)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.seed)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.offset)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.softmax_out_layout)},
        },
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    FlashAttentionScore, FlashAttentionScoreInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<FlashAttentionScoreInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FlashAttentionScoreInferShapeUtParam>);

} // namespace FlashAttentionScoreUT
