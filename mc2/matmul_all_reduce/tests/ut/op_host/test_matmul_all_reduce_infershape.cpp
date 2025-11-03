/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MatmulAllReduceInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceInfershape TearDown" << std::endl;
    }
};

TEST_F(MatmulAllReduceInfershape, infer_shape_for_2dim) {
    gert::StorageShape x1_shape = {{32, 64}, {}};
    gert::StorageShape x2_shape = {{64, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_3dim) {
    gert::StorageShape x1_shape = {{4, 8, 64}, {}};
    gert::StorageShape x2_shape = {{64, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_invalid_k) {
    gert::StorageShape x1_shape = {{32, 8}, {}};
    gert::StorageShape x2_shape = {{64, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_invalid_zero_k) {
    gert::StorageShape x1_shape = {{32, 0}, {}};
    gert::StorageShape x2_shape = {{0, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_3dim_quant_v4)
{
    gert::StorageShape x1_shape = {{4, 8, 64}, {}};
    gert::StorageShape x2_shape = {{64, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_add_rms_norm) {
    gert::StorageShape x1_shape = {{4, 8, 64}, {}};
    gert::StorageShape x2_shape = {{64, 128}, {}};
    gert::StorageShape bias_shape = {{128}, {}};
    gert::StorageShape residual_shape = {{4, 8, 128}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};
    gert::StorageShape output_shape_2 = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
            {residual_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape_1, ge::DT_FLOAT16, ge::FORMAT_ND},
            {output_shape_2, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_dtype) {
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType x2 = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 1)
        .InputDataTypes({&x1, &x2})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MatmulAllReduce")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}