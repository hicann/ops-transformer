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
#include <iostream>
#include <string>
#include <vector>

#include "quant_block_sparse_attn_host_ut_param.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

class QuantBlockSparseAttnInferDataTypeTest : public testing::TestWithParam<QuantBlockSparseAttnInferDTypeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantBlockSparseAttnInferDataTypeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantBlockSparseAttnInferDataTypeTest TearDown" << std::endl; }
};

TEST_P(QuantBlockSparseAttnInferDataTypeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    ge::DataType outputDtype0 = ge::DT_FLOAT;
    ge::DataType outputDtype1 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType(OP_NAME)
                             .NodeIoNum(16, 2)
                             .OutputDataTypes({&outputDtype0, &outputDtype1})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl(OP_NAME.c_str())->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0),
                  param.expectAttentionOutDtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
                  param.expectSoftmaxLseDtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnInferDataType, QuantBlockSparseAttnInferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnInferDTypeUtParam>);

} // namespace QuantBlockSparseAttnUT
