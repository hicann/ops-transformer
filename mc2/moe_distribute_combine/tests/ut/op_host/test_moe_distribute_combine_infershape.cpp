/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeCombineInferShapeUT {

class MoeDistributeCombineInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineInfershape TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineInfershape, infer_shape_0) {
    gert::StorageShape expand_x_shape = {{576, 7168}, {}};
    gert::StorageShape expert_ids_shape = {{32, 8}, {}};
    gert::StorageShape expand_idx_shape = {{32*8}, {}};
    gert::StorageShape ep_send_counts_shape = {{288}, {}};
    gert::StorageShape tp_send_counts_shape = {{2}, {}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeDistributeCombine",
        {
            {expand_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {expert_ids_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expand_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
            {ep_send_counts_shape, ge::DT_INT32, ge::FORMAT_ND},
            {tp_send_counts_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );
    std::vector<std::vector<int64_t>> x_output_shape = {{32, 7168},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, x_output_shape);
}
TEST_F(MoeDistributeCombineInfershape, infer_dtype_0) {
    ge::DataType expand_x_type = ge::DT_FLOAT16;
    ge::DataType expert_ids_type = ge::DT_INT32;
    ge::DataType expand_idx_type = ge::DT_INT32;
    ge::DataType ep_send_counts_type = ge::DT_INT32;
    ge::DataType tp_send_counts_type = ge::DT_INT32;
    ge::DataType expert_scales_type = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(6, 1)
        .InputDataTypes({&expand_x_type, &expert_ids_type, &expand_idx_type,
                         &ep_send_counts_type, &tp_send_counts_type, &expert_scales_type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeCombine")->infer_datatype;
    /* do infershape */
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

} //moe_distribute_combine_infershape_ut