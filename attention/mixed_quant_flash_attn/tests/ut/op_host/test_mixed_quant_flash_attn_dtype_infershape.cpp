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
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MixedQuantFlashAttnUT {

class MixedQuantFlashAttnInferDTypeTest : public testing::TestWithParam<MixedQuantFlashAttnInferDTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MixedQuantFlashAttn InferDTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MixedQuantFlashAttn InferDTypeTest TearDown" << std::endl;
    }
};

TEST_P(MixedQuantFlashAttnInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<void*> inputDataTypes;
    inputDataTypes.emplace_back(&param.q_dtype);
    inputDataTypes.emplace_back(&param.k_dtype);
    inputDataTypes.emplace_back(&param.v_dtype);

    ge::DataType attn_out_dtype_init = ge::DT_UNDEFINED;
    ge::DataType softmax_lse_dtype_init = ge::DT_UNDEFINED;
    std::vector<void*> outputDataTypes;
    outputDataTypes.emplace_back(&attn_out_dtype_init);
    outputDataTypes.emplace_back(&softmax_lse_dtype_init);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("MixedQuantFlashAttn")
            .IrInstanceNum({1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 1})
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs(std::vector<std::pair<std::string, Ops::Transformer::AnyValue>>{
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
            })
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MixedQuantFlashAttn")->infer_datatype;
    auto* dtypeContext = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(dtypeContext), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(dtypeContext->GetOutputDataType(0), param.expect_attn_out_dtype);
        if (param.expect_softmax_lse_dtype != ge::DT_UNDEFINED) {
            EXPECT_EQ(dtypeContext->GetOutputDataType(1), param.expect_softmax_lse_dtype);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MixedQuantFlashAttn, MixedQuantFlashAttnInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<MixedQuantFlashAttnInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MixedQuantFlashAttnInferDTypeUtParam>);

} // namespace MixedQuantFlashAttnUT
