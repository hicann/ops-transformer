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
#include "stem_oam_prep_varlen_q_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace StemOamPrepVarlenQUT {

class StemPrepQInferDTypeTest : public testing::TestWithParam<StemPrepQInferDTypeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "StemOamPrepVarlenQ InferDTypeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StemOamPrepVarlenQ InferDTypeTest TearDown" << std::endl; }
};

TEST_P(StemPrepQInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<void *> inputDataTypes;
    inputDataTypes.emplace_back(&param.q_dtype);

    ge::DataType qFlat_dtype_init = ge::DT_UNDEFINED;
    std::vector<void *> outputDataTypes;
    outputDataTypes.emplace_back(&qFlat_dtype_init);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("StemOamPrepVarlenQ")
            .IrInstanceNum({1, 1, 1, 0}, {1})
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs({
                {"stemBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stemBlockSize)},
                {"stemStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stem_stride)},
            })
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("StemOamPrepVarlenQ")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0),
                  param.expect_qFlat_dtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StemOamPrepVarlenQ, StemPrepQInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<StemPrepQInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StemPrepQInferDTypeUtParam>);

} // namespace StemOamPrepVarlenQUT
