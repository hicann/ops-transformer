/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cstdlib>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "mc2_tiling_case_executor.h"
#include "../test_moe_distribute_dispatch_v2_host_ut_param.h"

namespace MoeDistributeDispatchV2UT {

class MoeDistributeDispatchV2Arch35TilingTest : public testing::TestWithParam<MoeDistributeDispatchV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Arch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Arch35TilingTest TearDown" << std::endl;
    }
};

TEST_P(MoeDistributeDispatchV2Arch35TilingTest, param)
{
    auto param = GetParam();
    struct MoeDistributeDispatchV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatchV2",
        {param.x, param.expert_ids, param.scales, param.x_active_mask, param.expert_scales, param.elastic_info,
         param.performance_info},
        {param.expand_x, param.dynamic_scales, param.assist_info_for_combine, param.expert_token_nums,
         param.ep_recv_count, param.tp_recv_count, param.expand_scales},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group_ep)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_rank_id)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moe_expert_num)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group_tp)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.tp_world_size)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.tp_rank_id)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expert_shard_type)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.shared_expert_num)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.shared_expert_rank_num)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.global_bs)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expert_token_nums_type)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_alg)},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.zero_expert_num)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.copy_expert_num)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.const_expert_num)}},
        &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.ranksize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(
    MoeDistributeDispatchV2, MoeDistributeDispatchV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MoeDistributeDispatchV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MoeDistributeDispatchV2TilingUtParam>);

TEST_F(MoeDistributeDispatchV2Arch35TilingTest, A5SmallWindowCcuFailed)
{
    const char* envValue = getenv("HCCL_BUFFSIZE");
    std::string originalEnv = envValue != nullptr ? envValue : "";
    setenv("HCCL_BUFFSIZE", "1", 1);

    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatchV2",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 7}, {16, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{896, 7168}, {896, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{896}, {896}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(56)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "3510", 20, 196608);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);

    if (originalEnv.empty()) {
        unsetenv("HCCL_BUFFSIZE");
    } else {
        setenv("HCCL_BUFFSIZE", originalEnv.c_str(), 1);
    }
}

} // namespace MoeDistributeDispatchV2UT
