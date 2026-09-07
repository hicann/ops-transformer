/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeUpdateExpertInfershapeUT {
class MoeUpdateExpertInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeUpdateExpertInfershape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeUpdateExpertInfershape TearDown" << std::endl;
    }
};

TEST_F(MoeUpdateExpertInfershape, MoeUpdateExpertTestShape)
{
    gert::StorageShape expertIdsShape = {{128, 8}, {128, 8}};
    gert::StorageShape eplbTableShape = {{56, 5}, {56, 5}};

    gert::StorageShape balancedExpertIdsShape = {{128, 8}, {128, 8}};
    gert::StorageShape balancedActiveMaskShape = {{128, 8}, {128, 8}};

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "MoeUpdateExpert",
        {
            {expertIdsShape, ge::DT_INT64, ge::FORMAT_ND},
            {eplbTableShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {balancedExpertIdsShape, ge::DT_INT64, ge::FORMAT_ND},
            {balancedActiveMaskShape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {"localRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(localRankId)},
            {"worldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(worldSize)},
            {"balanceMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(balanceMode)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{128, 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MoeUpdateExpertInfershape, MoeUpdateExpertTestEnhancedShape)
{
    gert::StorageShape expertIdsShape = {{128, 8}, {128, 8}};
    gert::StorageShape eplbTableShape = {{56, 5}, {56, 5}};
    gert::StorageShape expertScalesShape = {{128, 8}, {128, 8}};
    gert::StorageShape pruingThresholdShape = {{
                                                   8,
                                               },
                                               {
                                                   8,
                                               }};
    gert::StorageShape activeMaskShape = {{
                                              128,
                                          },
                                          {
                                              128,
                                          }};

    gert::StorageShape balancedExpertIdsShape = {{128, 8}, {128, 8}};
    gert::StorageShape balancedActiveMaskShape = {{128, 8}, {128, 8}};

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "MoeUpdateExpert",
        {
            {expertIdsShape, ge::DT_INT64, ge::FORMAT_ND},
            {eplbTableShape, ge::DT_INT32, ge::FORMAT_ND},
            {expertScalesShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {pruingThresholdShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {activeMaskShape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {balancedExpertIdsShape, ge::DT_INT64, ge::FORMAT_ND},
            {balancedActiveMaskShape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {"localRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(localRankId)},
            {"worldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(worldSize)},
            {"balanceMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(balanceMode)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{128, 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MoeUpdateExpertInfershape, MoeUpdateExpertTestType)
{
    ge::DataType expertIdsType = ge::DT_INT64;
    ge::DataType eplbTableType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 2)
                             .InputDataTypes({&expertIdsType, &eplbTableType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeUpdateExpert")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_BOOL);
}

TEST_F(MoeUpdateExpertInfershape, MoeUpdateExpertTestEnhancedType)
{
    ge::DataType expertIdsType = ge::DT_INT64;
    ge::DataType eplbTableType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_FLOAT16;
    ge::DataType pruningThresholdType = ge::DT_FLOAT;
    ge::DataType activeMaskType = ge::DT_BOOL;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(5, 2)
            .InputDataTypes({&expertIdsType, &eplbTableType, &expertScalesType, &pruningThresholdType, &activeMaskType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeUpdateExpert")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_BOOL);
}
} // namespace MoeUpdateExpertInfershapeUT
