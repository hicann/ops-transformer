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
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#define private public
#include "platform/platform_info.h"
#undef private

namespace MoeDistributeDispatchV2 {
class MoeDistributeDispatchV2Infershape : public testing::Test {};

// infer shape with bias, success
TEST_F(MoeDistributeDispatchV2Infershape, inferShape0)
{
    gert::StorageShape expandXShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{expandXShape, ge::DT_INT32, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}},
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
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MoeDistributeDispatchV2Infershape, inferDtype0)
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
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .InputDataTypes({&expandXType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// ep_rank_id < shared_expert_rank_num：走共享专家卡分支（与 MoE 主分支区分）
TEST_F(MoeDistributeDispatchV2Infershape, inferShape_SharedExpertLowRankId)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expertOutputShape = {{256, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

// ============================================
// rtGetSocSpec mock 控制（在 mc2_tiling_ut_stub 中提供实现）
// ============================================
#include "rt_soc_spec_mocker.h"

// ============================================
// A2 平台补充测试
// ============================================

// 无共享专家：走 MoE 分支（epRankId >= sharedExpertRankNum）
// 覆盖 lines 236, 237
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_NoSharedExperts)
{
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 有 expertScales 输入：覆盖 epRecvCount 和 expandScales 的 expertScales 分支
// 覆盖 lines 283, 307
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_WithExpertScales)
{
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// globalBs < 0：globalBsReal 钳制为 -1，最终 a = -1
// 覆盖 line 251
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_NegativeGlobalBs)
{
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-99)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{-1, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// InferDtype MX 量化 + scales 输入：覆盖 InferDataTypeDynamicScales MX 分支
// 覆盖 line 120
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_MxQuant_WithScales)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 6)
                             .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                                         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                                         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                                         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                             .InputDataTypes({&xType, &expertIdsType, &scalesType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
}

// InferDtype NO_QUANT + scales：覆盖 quantFlag + NO_QUANT 分支
// 覆盖 line 123
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_NoQuant_WithScales)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 6)
                             .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                                         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                                         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                                         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                             .InputDataTypes({&xType, &expertIdsType, &scalesType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    // quantFlag=true, quantMode=0 → dynamicScalesDtype = scalesType = DT_FLOAT
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT);
}

// A2 packed INT4 uses INT32 x plus a required per-token FP32 scale. The scale
// is transported with x and must not cause expand_x to be inferred as INT8.
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A2_Int32NoQuant_WithScales)
{
    ge::DataType xType = ge::DT_INT32;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 6)
                             .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                                         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                                         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
                                         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                             .InputDataTypes({&xType, &expertIdsType, &scalesType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT);
}

// ============================================
// A5 平台 / rtGetSocSpec 失败场景测试
// ============================================

// rtGetSocSpec 失败 + elasticInfo
// 覆盖 lines 77-78, 88-89, 239-242, 288-289
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_RtGetSocSpecFail_WithElasticInfo)
{
    SetRtSocSpecFail(true);

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecFail(false);
}

// A5 平台 InferShape PERGROUP 量化 + elasticInfo
// 覆盖 lines 95-101, 262, 239-242
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_PerGroupQuant)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape MX 量化 + elasticInfo
// 覆盖 lines 102-105
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_MxQuant)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape NO_QUANT + scalesShape + elasticInfo（v2 特有分支）
// 覆盖 lines 106-109
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_NoQuant_WithScales)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape scalesShape = {{32, 56}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape else 分支（quantMode=1 走 else）+ elasticInfo
// 覆盖 lines 111-112
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_StaticQuant_Else)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape 共享专家 + elasticInfo → isValidShared 分支
// 覆盖 lines 244-247
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_SharedExperts_ElasticInfo)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape 非 A2 + expertScales + tp≠2 → 覆盖 v2 特有 epRecvCount 分支
// 覆盖 lines 290-291
TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A5_ExpertScales_NonA2EpRecvCount)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType STATIC + yDtype=DT_INT8
// 覆盖 lines 315, 317, 318, 353, 362-367
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_StaticQuant_Int8)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs(
                {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                 {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                 {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                 {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT8);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType PERTOKEN + yDtype=DT_FLOAT8_E4M3FN
// 覆盖 lines 322, 323
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_PerTokenQuant_E4M3)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype",
                         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType PERGROUP + yDtype=DT_FLOAT8_E4M3FN
// 覆盖 lines 329, 330
TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_PerGroupQuant_E4M3)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs({{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype",
                         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

static void SetupInfershapePlatform910B()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_InvalidEpRankId)
{
    SetupInfershapePlatform910B();
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_InvalidSharedExpertRankNum)
{
    SetupInfershapePlatform910B();
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_InvalidSharedSetting)
{
    SetupInfershapePlatform910B();
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_MoeRankNumNonPositive)
{
    SetupInfershapePlatform910B();
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_InvalidZeroBatch)
{
    SetupInfershapePlatform910B();
    gert::StorageShape xShape = {{0, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MoeDistributeDispatchV2Infershape, InferShape_A3_NoExpertScalesEpRecv)
{
    SetRtSocSpecShortVersion("Ascend910_93");
    SetRtSocSpecNpuArch("");
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910_93";
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV2",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_BadYdtypeStatic)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs(
                {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                 {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                 {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                 {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_BadYdtypePertoken)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs(
                {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                 {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                 {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                 {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

TEST_F(MoeDistributeDispatchV2Infershape, InferDtype_A5_BadYdtypeMx)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 6)
            .NodeAttrs(
                {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                 {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
                 {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                 {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                 {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))}})
            .InputDataTypes({&xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2");
    ASSERT_NE(opImpl, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

} // namespace MoeDistributeDispatchV2
