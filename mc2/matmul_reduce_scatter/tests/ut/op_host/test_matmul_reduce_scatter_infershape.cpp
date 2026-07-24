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

class MatmulReduceScatterInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterInfershape TearDown" << std::endl;
    }
};

TEST_F(MatmulReduceScatterInfershape, Basic)
{
    gert::StorageShape x1StorageShape = {{8192, 1536}, {}};
    gert::StorageShape x2StorageShape = {{1536, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatter",
        {
            {x1StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulReduceScatterInfershape, EmptyTensorTest)
{
    gert::StorageShape x1StorageShape = {{8192, 0}, {}};
    gert::StorageShape x2StorageShape = {{0, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatter",
        {
            {x1StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulReduceScatterInfershape, InferDataType_Float16)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 1)
                             .InputDataTypes({&x1Type, &x2Type})
                             .OutputDataTypes({&outputType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MatmulReduceScatter");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
}

TEST_F(MatmulReduceScatterInfershape, InferDataType_Bfloat16)
{
    ge::DataType x1Type = ge::DT_BF16;
    ge::DataType x2Type = ge::DT_BF16;
    ge::DataType outputType = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 1)
                             .InputDataTypes({&x1Type, &x2Type})
                             .OutputDataTypes({&outputType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MatmulReduceScatter");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
}