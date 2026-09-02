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
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace FlashAttentionScoreUT {

class FlashAttentionScoreInferDTypeTest : public testing::TestWithParam<FlashAttentionScoreInferDTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScore InferDTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScore InferDTypeTest TearDown" << std::endl;
    }
};

TEST_P(FlashAttentionScoreInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<uint32_t> inputInstance(22, 1);
    std::vector<uint32_t> outputInstance(4, 1);
    ge::DataType inputDtype = param.input_dtype;
    std::vector<void *> inputDataTypes = {&inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype,
                                          &inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype,
                                          &inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype, &inputDtype,
                                          &inputDtype, &inputDtype, &inputDtype, &inputDtype};
    std::vector<void *> outputDataTypes = {&param.softmaxMax_dtype, &param.softmaxSum_dtype, &param.softmaxOut_dtype,
                                           &param.attentionOut_dtype};

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("FlashAttentionScore")
            .IrInstanceNum(inputInstance, outputInstance)
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs(
                {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale_value)},
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
                 {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.softmax_out_layout)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.softmaxMax_dtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.softmaxSum_dtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2), param.softmaxOut_dtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3),
                  param.attentionOut_dtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FlashAttentionScore, FlashAttentionScoreInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<FlashAttentionScoreInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FlashAttentionScoreInferDTypeUtParam>);

} // namespace FlashAttentionScoreUT
