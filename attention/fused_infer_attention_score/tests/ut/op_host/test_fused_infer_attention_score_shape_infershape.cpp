/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "fused_infer_attention_score_param.h"
#include "infer_shape_case_executor.h"

namespace FusedInferAttentionScoreUT {

class FusedInferAttentionScoreInferShapeTest : public testing::TestWithParam<FusedInferAttentionInferShapeUtParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// 临时添加，待框架修复后删除
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FusedInferAttentionScoreInferShapeTest);

TEST_P(FusedInferAttentionScoreInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.query);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.key);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.value);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.pse_shift);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.atten_mask);
    if (param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.actual_seq_lengths);
    if (param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.actual_seq_lengths_kv);
    if (param.inputInstance[7] == 1)
        inputTensorDesc.emplace_back(param.dequant_scale1);
    if (param.inputInstance[8] == 1)
        inputTensorDesc.emplace_back(param.quant_scale1);
    if (param.inputInstance[9] == 1)
        inputTensorDesc.emplace_back(param.dequant_scale2);
    if (param.inputInstance[10] == 1)
        inputTensorDesc.emplace_back(param.quant_scale2);
    if (param.inputInstance[11] == 1)
        inputTensorDesc.emplace_back(param.quant_offset2);
    if (param.inputInstance[12] == 1)
        inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstance[13] == 1)
        inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstance[14] == 1)
        inputTensorDesc.emplace_back(param.block_table);
    if (param.inputInstance[15] == 1)
        inputTensorDesc.emplace_back(param.query_padding_size);
    if (param.inputInstance[16] == 1)
        inputTensorDesc.emplace_back(param.kv_padding_size);
    if (param.inputInstance[17] == 1)
        inputTensorDesc.emplace_back(param.key_antiquant_scale);
    if (param.inputInstance[18] == 1)
        inputTensorDesc.emplace_back(param.key_antiquant_offset);
    if (param.inputInstance[19] == 1)
        inputTensorDesc.emplace_back(param.value_antiquant_scale);
    if (param.inputInstance[20] == 1)
        inputTensorDesc.emplace_back(param.value_antiquant_offset);
    if (param.inputInstance[21] == 1)
        inputTensorDesc.emplace_back(param.key_shared_prefix);
    if (param.inputInstance[22] == 1)
        inputTensorDesc.emplace_back(param.value_shared_prefix);
    if (param.inputInstance[23] == 1)
        inputTensorDesc.emplace_back(param.actual_shared_prefix_len);
    if (param.inputInstance[24] == 1)
        inputTensorDesc.emplace_back(param.query_rope);
    if (param.inputInstance[25] == 1)
        inputTensorDesc.emplace_back(param.key_rope);
    if (param.inputInstance[26] == 1)
        inputTensorDesc.emplace_back(param.key_rope_antiquant_scale);
    if (param.inputInstance[27] == 1)
        inputTensorDesc.emplace_back(param.dequant_scale_query);
    if (param.inputInstance[28] == 1)
        inputTensorDesc.emplace_back(param.learnable_sink);
    if (param.inputInstance[29] == 1)
        inputTensorDesc.emplace_back(param.q_start_idx);
    if (param.inputInstance[30] == 1)
        inputTensorDesc.emplace_back(param.kv_start_idx);

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.attention_out);
    if (param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.softmax_lse);

    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore", inputTensorDesc, outputTensorDesc,
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_heads)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tokens)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tokens)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_key_value_heads)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_mode)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.softmax_lse_flag)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.key_antiquant_mode)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.value_antiquant_mode)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.query_quant_mode)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)}},
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    FusedInferAttentionScore, FusedInferAttentionScoreInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<FusedInferAttentionInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FusedInferAttentionInferShapeUtParam>);

} // namespace FusedInferAttentionScoreUT
