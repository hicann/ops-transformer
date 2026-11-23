/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"
#include "../../../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h"

namespace {
using Attr = gert::TilingContextPara::OpAttr;
using Tensor = gert::TilingContextPara::TensorDescription;

std::vector<Attr> GetRoceDispatchAttrs(int64_t ranksPerServer, int64_t windowBytes)
{
    const std::vector<std::pair<std::string, int64_t>> values = {
        {"ep_world_size", 8},          {"ep_rank_id", 0}, {"moe_expert_num", 16},   {"ccl_buffer_size", windowBytes},
        {"tp_world_size", 1},          {"tp_rank_id", 0}, {"expert_shard_type", 0}, {"shared_expert_num", 0},
        {"shared_expert_rank_num", 0}, {"quant_mode", 0}, {"global_bs", 0},         {"expert_token_nums_type", 1}};
    std::vector<Attr> attrs;
    for (const std::pair<std::string, int64_t>& value : values) {
        attrs.emplace_back(value.first, Ops::Transformer::AnyValue::CreateFrom<int64_t>(value.second));
    }
    attrs.emplace_back("comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("hierarchy"));
    for (const char* name : {"zero_expert_num", "copy_expert_num", "const_expert_num"}) {
        attrs.emplace_back(name, Ops::Transformer::AnyValue::CreateFrom<int64_t>(0));
    }
    attrs.emplace_back("y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16));
    attrs.emplace_back("rank_size_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ranksPerServer));
    return attrs;
}

void RunRoceDispatchCase(int64_t ranksPerServer, int64_t assistSize, int64_t windowBytes, ge::graphStatus expected)
{
    std::vector<Tensor> inputs = {{{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{16, 1024}, {16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{}, ge::DT_BOOL, ge::FORMAT_ND},
                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{}, ge::DT_INT64, ge::FORMAT_ND}};
    std::vector<Tensor> outputs = {{{{256, 1024}, {256, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                   {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   {{{assistSize}, {assistSize}}, ge::DT_INT32, ge::FORMAT_ND},
                                   {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                   {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                   {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                   {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}};
    struct MoeDistributeDispatchCompileInfo {
    } compileInfo;
    gert::TilingContextPara params("MoeDistributeDispatchV3", inputs, outputs,
                                   GetRoceDispatchAttrs(ranksPerServer, windowBytes), &compileInfo, "3510", 48,
                                   256U * 1024U);
    Mc2ExecuteTestCase(params, {{"rankNum", 8}}, expected, 1344U);
}

TEST(MoeDistributeDispatchV3RoceTiling, Topk16AndCompactAssist)
{
    RunRoceDispatchCase(4, 836, 64U * 1024U * 1024U, ge::GRAPH_SUCCESS);
}

TEST(MoeDistributeDispatchV3RoceTiling, RejectsSingleServer)
{
    RunRoceDispatchCase(8, 836, 64U * 1024U * 1024U, ge::GRAPH_FAILED);
}

TEST(MoeDistributeDispatchV3RoceTiling, RejectsTruncatedSharedCount)
{
    RunRoceDispatchCase(4, 835, 64U * 1024U * 1024U, ge::GRAPH_FAILED);
}

TEST(MoeDistributeDispatchV3RoceTiling, RejectsInvalidTopologyAndSmallWindow)
{
    RunRoceDispatchCase(3, 836, 64U * 1024U * 1024U, ge::GRAPH_FAILED);
    RunRoceDispatchCase(4, 836, 1024U * 1024U, ge::GRAPH_FAILED);
}

TEST(MoeDistributeDispatchV3RoceTiling, AssistIncludesCountAndInt32TailPadding)
{
    EXPECT_EQ(Mc2Tiling::GetRoceAssistInfoSize(256, 8, 16), 836U);
    EXPECT_EQ(Mc2Tiling::GetRoceAssistInfoSize(1, 3, 17), 53U);
}
} // namespace
