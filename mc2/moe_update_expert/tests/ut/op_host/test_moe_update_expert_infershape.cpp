/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeUpdateExpertInfershapeUT{
class MoeUpdateExpertInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MoeUpdateExpertInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MoeUpdateExpertInfershape TearDown" << std::endl; }
};

TEST_F(MoeUpdateExpertInfershape, moe_update_expert_test_shape) {
    gert::StorageShape expert_ids_shape = {{128, 8}, {128, 8}};
    gert::StorageShape eplb_table_shape = {{56, 5}, {56, 5}};

    gert::StorageShape balanced_expert_ids_shape = {{128, 8}, {128, 8}};
    gert::StorageShape balanced_active_mask_shape = {{128, 8}, {128, 8}};

    int64_t local_rank_id = 0;
    int64_t world_size = 8;
    int64_t balance_mode = 0;

    gert::InfershapeContextPara infershapeContextPara("MoeUpdateExpert",
        {
            {expert_ids_shape, ge::DT_INT64, ge::FORMAT_ND},
            {eplb_table_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {balanced_expert_ids_shape, ge::DT_INT64, ge::FORMAT_ND},
            {balanced_active_mask_shape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(local_rank_id)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(world_size)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(balance_mode)},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{128, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MoeUpdateExpertInfershape, moe_update_expert_test_enhanced_shape) {
    gert::StorageShape expert_ids_shape = {{128, 8}, {128, 8}};
    gert::StorageShape eplb_table_shape = {{56, 5}, {56, 5}};
    gert::StorageShape expert_scales_shape = {{128, 8}, {128, 8}};
    gert::StorageShape pruing_threshold_shape = {{8, }, {8, }};
    gert::StorageShape active_mask_shape = {{128,}, {128,}};

    gert::StorageShape balanced_expert_ids_shape = {{128, 8}, {128, 8}};
    gert::StorageShape balanced_active_mask_shape = {{128, 8}, {128, 8}};

    int64_t local_rank_id = 0;
    int64_t world_size = 8;
    int64_t balance_mode = 0;

    gert::InfershapeContextPara infershapeContextPara("MoeUpdateExpert",
        {
            {expert_ids_shape, ge::DT_INT64, ge::FORMAT_ND},
            {eplb_table_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {pruing_threshold_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {active_mask_shape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {balanced_expert_ids_shape, ge::DT_INT64, ge::FORMAT_ND},
            {balanced_active_mask_shape, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(local_rank_id)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(world_size)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(balance_mode)},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{128, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}


TEST_F(MoeUpdateExpertInfershape, moe_update_expert_test_type) {
    ge::DataType expert_ids_type = ge::DT_INT64;
    ge::DataType eplb_table_type = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(2, 2)
                    .InputDataTypes({&expert_ids_type, &eplb_table_type})
                    .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeUpdateExpert")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_BOOL);
}

TEST_F(MoeUpdateExpertInfershape, moe_update_expert_test_enhanced_type) {
    ge::DataType expert_ids_type = ge::DT_INT64;
    ge::DataType eplb_table_type = ge::DT_INT32;
    ge::DataType expert_scales_type = ge::DT_FLOAT16;
    ge::DataType pruning_threshold_type = ge::DT_FLOAT;
    ge::DataType active_mask_type = ge::DT_BOOL;

    auto contextHolder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(5, 2)
                    .InputDataTypes({&expert_ids_type, &eplb_table_type, &expert_scales_type,
                                     &pruning_threshold_type, &active_mask_type})
                    .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeUpdateExpert")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_BOOL);
}
}