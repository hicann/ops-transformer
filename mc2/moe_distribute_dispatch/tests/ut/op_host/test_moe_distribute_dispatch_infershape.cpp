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
#define private public
#include "platform/platform_info.h"

namespace MoeDistributeDispatchInferShapeUT {

class MoeDistributeDispatchInfershape : public testing::Test {
};

namespace
{
    bool Comp(const gert::Shape *x, const gert::Shape y) {
        for (int i = 0; i < x->GetDimNum(); i++) {
            std::cout << x->GetDim(i) << " " << y.GetDim(i) << std::endl;
            if (x->GetDim(i) != y.GetDim(i)) {
                return false;
            }
        }
        return true;
    }
} // namespace

// infer shape with bias, success
TEST_F(MoeDistributeDispatchInfershape, infer_shape_0) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    gert::StorageShape expand_x_shape = {{32, 7168}, {}};
    gert::StorageShape expert_ids_shape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeDistributeDispatch",
        {
            {expand_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {expert_ids_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expand_x_output_shape = {{576, 7168}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expand_x_output_shape);
}

TEST_F(MoeDistributeDispatchInfershape, infer_shape_1) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    gert::InfershapeContextPara infershapeContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expand_x_output_shape = {{576, 7168}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expand_x_output_shape);
}

TEST_F(MoeDistributeDispatchInfershape, infer_dtype_0) {
    ge::DataType expand_x_type = ge::DT_FLOAT16;
    ge::DataType expert_ids_type = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 6)
        .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                    {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                    {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                    {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                    {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                    {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
        .NodeInputTd(0, expand_x_type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, expert_ids_type, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&expand_x_type, &expert_ids_type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeDispatch")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

} //moe_distribute_dispatch_infershape_ut