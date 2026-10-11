/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#define private public
#include "platform/platform_info.h"

namespace MoeDistributeDispatchInferShapeUT {

class MoeDistributeDispatchInfershape : public testing::Test {};

namespace {
bool Comp(const gert::Shape* x, const gert::Shape y)
{
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
TEST_F(MoeDistributeDispatchInfershape, InferShape0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::StorageShape expandXShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatch",
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expandXOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expandXOutputShape);
}

TEST_F(MoeDistributeDispatchInfershape, InferShape1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatch",
        {{{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expandXOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expandXOutputShape);
}

TEST_F(MoeDistributeDispatchInfershape, InferDtype0)
{
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, expandXType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&expandXType, &expertIdsType})
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

} // namespace MoeDistributeDispatchInferShapeUT
