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
#include "mixed_quant_flash_attn_param.h"
#include "infer_shape_case_executor.h"

namespace MixedQuantFlashAttnUT {

class MixedQuantFlashAttnInferShapeTest : public testing::TestWithParam<MixedQuantFlashAttnInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MixedQuantFlashAttn InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MixedQuantFlashAttn InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(MixedQuantFlashAttnInferShapeTest, param)
{
    auto param = GetParam();

    gert::InfershapeContextPara::TensorDescription dummy({}, ge::DT_UNDEFINED, ge::FORMAT_NULL);

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    inputTensorDesc.emplace_back(param.q);         // 0: q
    inputTensorDesc.emplace_back(param.k);         // 1: k
    inputTensorDesc.emplace_back(param.v);         // 2: v
    inputTensorDesc.emplace_back(param.k_descale); // 3: k_descale
    inputTensorDesc.emplace_back(param.v_descale); // 4: v_descale
    inputTensorDesc.emplace_back(dummy);           // 5: block_table
    inputTensorDesc.emplace_back(dummy);           // 6: cu_seqlens_q
    inputTensorDesc.emplace_back(dummy);           // 7: seqused_q
    inputTensorDesc.emplace_back(dummy);           // 8: seqused_kv
    inputTensorDesc.emplace_back(dummy);           // 9: sinks
    inputTensorDesc.emplace_back(dummy);           // 10: attn_mask
    inputTensorDesc.emplace_back(dummy);           // 11: metadata

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    outputTensorDesc.emplace_back(param.attn_out);
    outputTensorDesc.emplace_back(param.softmax_lse);

    gert::InfershapeContextPara infershapeContextPara(
        "MixedQuantFlashAttn", inputTensorDesc, outputTensorDesc,
        {
            {"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_compute_mode)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_left)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_right)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
            {"layout_attn_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.lay_atten)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
        });

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    MixedQuantFlashAttn, MixedQuantFlashAttnInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<MixedQuantFlashAttnInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MixedQuantFlashAttnInferShapeUtParam>);

} // namespace MixedQuantFlashAttnUT
