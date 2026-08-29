/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_all_gather_matmul_v3_dtype_infershape.cpp
 * \brief inferdtype ut
 */

#include <gtest/gtest.h>
#include "test_all_gather_matmul_v3_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AllGatherMatmulV3UT {

class AllGatherMatmulV3InferDataTypeTest : public testing::TestWithParam<AllGatherMatmulV3InferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV3InferDataTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV3InferDataTypeTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulV3InferDataTypeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    // def 输入顺序: context(0) x1(1) x2(2) bias(3) x1_scale(4) x2_scale(5)，仅放入已实例化的输入
    ge::DataType contextDtype = ge::DT_INT32;
    std::vector<void *> inputDataTypes;
    inputDataTypes.emplace_back(&contextDtype);
    inputDataTypes.emplace_back(&param.x1);
    inputDataTypes.emplace_back(&param.x2);
    inputDataTypes.emplace_back(&param.x1Scale);
    inputDataTypes.emplace_back(&param.x2Scale);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("AllGatherMatmulV3")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
                        {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransA)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransB)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankSize)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
                        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV3")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.expectYDtype);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
                  param.expectGatherOutDtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV3InferDataTypeUT, AllGatherMatmulV3InferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV3InferDataTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV3InferDataTypeUtParam>);

} // namespace AllGatherMatmulV3UT
