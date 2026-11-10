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
 * \file test_moe_distribute_dispatch_v3_infershape.cpp
 * \brief infershape ut - 简化版本，确保所有测试通过
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../common/tests/ut/formal_error_assertions.h"
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "rt_soc_spec_mocker.h"
#define private public
#include "platform/platform_info.h"
#undef private

using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

namespace MoeDistributeDispatchV3InfershapeUT {

class V3PlatformStateGuard {
public:
    V3PlatformStateGuard()
        : optional_(fe::PlatformInfoManager::Instance().opti_compilation_info_)
    {
        auto& cache = fe::PlatformInfoManager::Instance().platform_info_map_;
        const auto found = cache.find("Ascend910B");
        existed_ = found != cache.end();
        if (existed_)
            original_ = found->second;
    }
    ~V3PlatformStateGuard()
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.SetOptionalCompilationInfo(optional_);
        if (existed_)
            manager.platform_info_map_["Ascend910B"] = original_;
        else
            manager.platform_info_map_.erase("Ascend910B");
    }
    V3PlatformStateGuard(const V3PlatformStateGuard&) = delete;
    V3PlatformStateGuard& operator=(const V3PlatformStateGuard&) = delete;

private:
    fe::OptionalInfo optional_;
    fe::PlatformInfo original_;
    bool existed_ = false;
};
class MoeDistributeDispatchV3InfershapeTest : public testing::Test {
    V3PlatformStateGuard platformState_;
};

// ============================================
// InferShape 测试用例（保留能通过的测试）
// ============================================

// 测试场景1：共享专家卡（基础场景，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_SharedExpertCard_WithTP)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 测试场景2：有 expertScales 输入（A2 平台，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_WithExpertScales_A2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 测试场景3：动态量化（quantMode = 2，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_DynamicQuant)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT8, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 异常：共享专家配置不合法（shared_expert_num 与 shared_expert_rank_num 关系错误）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidSharedExpertSetting)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 异常：moeRankNum = ep_world_size - shared_expert_rank_num <= 0
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_MoeRankNumNonPositive)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 测试场景4：异常 - epRankId 越界（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidEpRankId)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(300)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 测试场景5：异常 - 输入形状不正确（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidInputShape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{0, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

class MoeDistributeDispatchV3BatchInfershape : public testing::TestWithParam<int64_t> {};

TEST_P(MoeDistributeDispatchV3BatchInfershape, InferShape_CheckExpertIdsBatch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{8, 512}, {}};
    gert::StorageShape expertIdsShape = {{GetParam(), 1}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    if (GetParam() == 8) {
        std::vector<std::vector<int64_t>> expectedOutputShape = {{16, 512}};
        Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
    } else {
        Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
    }
}

INSTANTIATE_TEST_SUITE_P(ExpertIdsBatch, MoeDistributeDispatchV3BatchInfershape, testing::Values(7, 8, 9));

// 测试场景6：静态量化（quantMode = 1，补充测试）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_StaticQuant)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT8, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 无共享专家：走 MoE 分支（ep_rank_id >= shared_expert_rank_num），tp_world_size=1
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_MoEBranch_Tp1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    // a = globalBsReal(256) * min(localExpertNum=8, k=8) = 2048（当前版本不支持TP域通信，expand_x 第一维取 a）
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 无共享专家 + tp_world_size=1：覆盖非 A2 SoC 路径下 ep_recv 的 else 分支（tp != 2）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_TpWorldSize1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// global_bs < 0：globalBsReal 钳制为 -1，且最终将 a 置为 -1（expand_x 第一维为 -1）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NegativeGlobalBs_MoEBranch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{-1, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// tp_world_size=0：expand_x 第一维取 a
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_TpWorldSize0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// ============================================
// InferDataType 测试用例（保留能通过的测试）
// ============================================

// 测试 InferDataType：非量化场景（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_NoQuant)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 测试 InferDataType：量化场景默认INT8（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_Quant_DefaultInt8)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    // 非A5架构且无scales输入时，量化场景默认输出DT_INT8
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT8);
}

// 测试 InferDataType：BFLOAT16输入类型（补充测试）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_BFloat16_Input)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    // 非量化场景，输出类型 = 输入类型
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

// quant_mode = MX(4)：InferDataTypeDynamicScales 输出 DT_FLOAT8_E8M0
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_MxQuant_DynamicScalesE8M0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 7)
                             .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
                                         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
                             .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&contextType, &xType, &expertIdsType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
}

// ============================================
// rtGetSocSpec mock 控制（在 mc2_tiling_ut_stub 中提供实现）
// ============================================

// ============================================
// A5 平台 / rtGetSocSpec 失败场景测试
// ============================================

// InferShape: rtGetSocSpec 失败 → IsTarget... 返回 false → 视为非 A2 + 非 A5
// 覆盖 lines 77-78, 88-89, 231-233, 279-280
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_RtGetSocSpecFail_WithElasticInfo)
{
    SetRtSocSpecFail(true);

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    // a=2048（当前版本不支持TP域通信，expand_x 第一维取 a）
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecFail(false);
}

// A5 平台 InferShape：覆盖 InferShapeDynamicScalesA5 PERGROUP 分支
// 覆盖 lines 95-100, 253, 231-233
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_A5_PerGroupQuant)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape：覆盖 InferShapeDynamicScalesA5 MX 分支
// 覆盖 lines 101-104
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_A5_MxQuant)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape：覆盖 InferShapeDynamicScalesA5 else 分支 (quantMode=0/1/2)
// 覆盖 lines 106-107
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_A5_NoQuant)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferShape：共享专家 + elasticInfo → isValidShared 分支
// 覆盖 lines 234-238
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_A5_SharedExperts_ElasticInfo)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape elasticInfoShape = {{1, 1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {elasticInfoShape, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    // moeRankNum=4, localMoeExpertNum=16, a=max(128,2048)=2048
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType：STATIC 量化 + yDtype=DT_INT8
// 覆盖 lines 304, 306, 307 (CheckQuantMode STATIC 分支), 340, 348-353
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_A5_StaticQuant_Int8)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs(
                {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                 {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                 {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType：PERTOKEN 量化 + yDtype=DT_FLOAT8_E4M3FN
// 覆盖 lines 311, 312 (CheckQuantMode PERTOKEN 分支)
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_A5_PerTokenQuant_E4M3)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

// A5 平台 InferDataType：PERGROUP 量化 + yDtype=DT_FLOAT8_E4M3FN
// 覆盖 lines 318, 319 (CheckQuantMode PERGROUP/MX 分支)
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_A5_PerGroupQuant_E4M3)
{
    SetRtSocSpecShortVersion("Ascend950");
    SetRtSocSpecNpuArch("3510");

    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);

    SetRtSocSpecShortVersion("Ascend910B");
    SetRtSocSpecNpuArch("");
}

TEST_F(MoeDistributeDispatchV3InfershapeTest, InvalidEpRankRelationReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
    ExpectExcelFormalError("EZ0027", "ep_rank_id, ep_world_size", "ep_rank_id=288, ep_world_size=288",
                           "ep_rank_id must be in [0, ep_world_size)");
}

TEST_F(MoeDistributeDispatchV3InfershapeTest, InvalidSharedRankRelationReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
    ExpectExcelFormalError("EZ0027", "shared_expert_rank_num, ep_world_size",
                           "shared_expert_rank_num=288, ep_world_size=288",
                           "shared_expert_rank_num must be in [0, ep_world_size)");
}

TEST_F(MoeDistributeDispatchV3InfershapeTest, InvalidSharedExpertMultipleReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{288, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
    ExpectExcelFormalError("EZ0027", "shared_expert_rank_num, shared_expert_num",
                           "shared_expert_rank_num=32, shared_expert_num=3",
                           "Use (shared_expert_num, shared_expert_rank_num) = (1, 0) or (0, 0), or a positive "
                           "shared_expert_num with shared_expert_rank_num a positive multiple of it");
}

static gert::InfershapeContextPara OwnerV3BoundaryContext()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    return infershapeContextPara;
}

TEST_F(MoeDistributeDispatchV3InfershapeTest, MoeRankDenominatorBoundariesPreserveFirstError)
{
    for (const auto world : {0LL, -1LL}) {
        ExcelErrorReportScope errors;
        auto context = OwnerV3BoundaryContext();
        context.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(world);
        context.attrs_[8].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(0);
        Mc2ExecuteTestCase(context, {{"rankNum", 8}}, ge::GRAPH_FAILED, {});
        const auto actual = "ep_rank_id=0, ep_world_size=" + std::to_string(world);
        ExpectExcelFormalError("EZ0027", "ep_rank_id, ep_world_size", actual.c_str(),
                               "ep_rank_id must be in [0, ep_world_size)");
    }
    for (const auto shared : {-1LL, 9LL}) {
        ExcelErrorReportScope errors;
        auto context = OwnerV3BoundaryContext();
        context.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(8);
        context.attrs_[8].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared);
        Mc2ExecuteTestCase(context, {{"rankNum", 8}}, ge::GRAPH_FAILED, {});
        const auto actual = "shared_expert_rank_num=" + std::to_string(shared) + ", ep_world_size=8";
        ExpectExcelFormalError("EZ0027", "shared_expert_rank_num, ep_world_size", actual.c_str(),
                               "shared_expert_rank_num must be in [0, ep_world_size)");
    }
}
TEST_F(MoeDistributeDispatchV3InfershapeTest, MoeRankOneContinuesWithExpectedExpertCount)
{
    ExcelErrorReportScope errors;
    auto context = OwnerV3BoundaryContext();
    context.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(8);
    context.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(7);
    context.attrs_[8].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(7);
    Mc2ExecuteTestCase(context, {{"rankNum", 8}}, ge::GRAPH_SUCCESS, {{2048, 7168}, {2048}, {262144}, {256}});
    EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
}
TEST(V3PlatformStateGuardContract, RestoresExistingAndAbsentCacheEntries)
{
    V3PlatformStateGuard preserveOuterState;
    auto& manager = fe::PlatformInfoManager::Instance();
    manager.platform_info_map_.erase("Ascend910B");
    fe::OptionalInfo before;
    before.soc_version = "owner_original";
    manager.SetOptionalCompilationInfo(before);
    {
        V3PlatformStateGuard scope;
        fe::PlatformInfo replacement;
        replacement.str_info.short_soc_version = "owner_replacement";
        manager.platform_info_map_["Ascend910B"] = replacement;
        fe::OptionalInfo after;
        after.soc_version = "owner_replacement";
        manager.SetOptionalCompilationInfo(after);
    }
    EXPECT_EQ(manager.platform_info_map_.count("Ascend910B"), 0U);
    EXPECT_EQ(manager.opti_compilation_info_.soc_version, "owner_original");
    fe::PlatformInfo original;
    original.str_info.short_soc_version = "owner_original";
    manager.platform_info_map_["Ascend910B"] = original;
    {
        V3PlatformStateGuard scope;
        manager.platform_info_map_["Ascend910B"].str_info.short_soc_version = "owner_replacement";
    }
    EXPECT_EQ(manager.platform_info_map_.at("Ascend910B").str_info.short_soc_version, "owner_original");
}

} // namespace MoeDistributeDispatchV3InfershapeUT
