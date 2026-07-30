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
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace StemIndexerUT {

class StemIndexerInferDTypeTest : public testing::TestWithParam<StemIndexerInferDTypeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "StemIndexer InferDTypeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StemIndexer InferDTypeTest TearDown" << std::endl; }
};

TEST_P(StemIndexerInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<void *> inputDataTypes;
    if (param.inputInstance[0] == 1)
        inputDataTypes.emplace_back(&param.qflat_dtype);
    if (param.inputInstance[1] == 1)
        inputDataTypes.emplace_back(&param.kflat_dtype);
    if (param.inputInstance[2] == 1)
        inputDataTypes.emplace_back(&param.vbias_dtype);
    if (param.inputInstance[3] == 1)
        inputDataTypes.emplace_back(&param.q_seq_lens_dtype);
    if (param.inputInstance[4] == 1)
        inputDataTypes.emplace_back(&param.kv_seq_lens_dtype);
    if (param.inputInstance[5] == 1)
        inputDataTypes.emplace_back(&param.num_prompt_tokens_dtype);
    if (param.inputInstance[6] == 1)
        inputDataTypes.emplace_back(&param.metadata_dtype);

    std::vector<void *> outputDataTypes;
    if (param.outputInstance[0] == 1)
        outputDataTypes.emplace_back(&param.sparse_indices_dtype);
    if (param.outputInstance[1] == 1)
        outputDataTypes.emplace_back(&param.sparse_seq_len_dtype);

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("StemIndexer")
                             .IrInstanceNum(param.inputInstance, param.outputInstance)
                             .InputDataTypes(inputDataTypes)
                             .OutputDataTypes(outputDataTypes)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("StemIndexer");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0),
                  param.sparse_indices_dtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
                  param.sparse_seq_len_dtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StemIndexer, StemIndexerInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<StemIndexerInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StemIndexerInferDTypeUtParam>);

} // namespace StemIndexerUT
