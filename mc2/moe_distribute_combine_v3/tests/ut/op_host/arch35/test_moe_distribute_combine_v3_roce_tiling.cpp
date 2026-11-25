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

namespace {
using Attr = gert::TilingContextPara::OpAttr;
using Tensor = gert::TilingContextPara::TensorDescription;

// 256 route rows and two UB domains with 16 tokens per rank.
constexpr int64_t MIN_ASSIST_INFO_SIZE = 816;

std::vector<Attr> GetRoceCombineAttrs(bool preReduce, int64_t quantMode, int64_t ranksPerServer)
{
    const std::vector<std::pair<std::string, int64_t>> values = {{"ep_world_size", 8},
                                                                 {"ep_rank_id", 0},
                                                                 {"moe_expert_num", 16},
                                                                 {"ccl_buffer_size", 64U * 1024U * 1024U},
                                                                 {"tp_world_size", 1},
                                                                 {"tp_rank_id", 0},
                                                                 {"expert_shard_type", 0},
                                                                 {"shared_expert_num", 0},
                                                                 {"shared_expert_rank_num", 0},
                                                                 {"global_bs", 0},
                                                                 {"out_dtype", 0},
                                                                 {"comm_quant_mode", quantMode},
                                                                 {"group_list_type", 0}};
    std::vector<Attr> attrs;
    for (const std::pair<std::string, int64_t>& value : values) {
        attrs.emplace_back(value.first, Ops::Transformer::AnyValue::CreateFrom<int64_t>(value.second));
    }
    attrs.emplace_back("comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("hierarchy"));
    for (const char* name : {"zero_expert_num", "copy_expert_num", "const_expert_num"}) {
        attrs.emplace_back(name, Ops::Transformer::AnyValue::CreateFrom<int64_t>(0));
    }
    attrs.emplace_back("enable_pre_reduce", Ops::Transformer::AnyValue::CreateFrom<bool>(preReduce));
    attrs.emplace_back("rank_size_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ranksPerServer));
    return attrs;
}

void RunRoceCombineCase(bool preReduce, int64_t quantMode, int64_t ranksPerServer, int64_t assistSize,
                        int64_t expandScalesSize, ge::graphStatus expected)
{
    std::vector<Tensor> inputs = {{{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{256, 1024}, {256, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{assistSize}, {assistSize}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{}, ge::DT_BOOL, ge::FORMAT_ND},
                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{}, ge::DT_INT64, ge::FORMAT_ND}};
    if (expandScalesSize > 0) {
        inputs.emplace_back(gert::StorageShape({expandScalesSize}, {expandScalesSize}), ge::DT_FLOAT, ge::FORMAT_ND);
    } else {
        inputs.push_back({{}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    inputs.insert(inputs.end(), {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 {{}, ge::DT_INT32, ge::FORMAT_ND},
                                 {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 {{}, ge::DT_INT64, ge::FORMAT_ND}});
    std::vector<Tensor> outputs = {{{{16, 1024}, {16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    struct MoeDistributeCombineV3CompileInfo {
    } compileInfo;
    gert::TilingContextPara params("MoeDistributeCombineV3", inputs, outputs,
                                   GetRoceCombineAttrs(preReduce, quantMode, ranksPerServer), &compileInfo, "3510", 48,
                                   256U * 1024U);
    const uint64_t quantKey = quantMode == 0 ? 0U : static_cast<uint64_t>(quantMode - 1);
    Mc2ExecuteTestCase(params, {{"rankNum", 8}}, expected, 140U + quantKey);
}

TEST(MoeDistributeCombineV3RoceTiling, PreReduceAndForwardQuantModes)
{
    for (bool preReduce : {true, false}) {
        for (int64_t quantMode : {0, 2, 3, 4}) {
            RunRoceCombineCase(preReduce, quantMode, 4, MIN_ASSIST_INFO_SIZE, 256, ge::GRAPH_SUCCESS);
        }
    }
}

TEST(MoeDistributeCombineV3RoceTiling, RejectsSingleServer)
{
    RunRoceCombineCase(true, 0, 8, MIN_ASSIST_INFO_SIZE, 256, ge::GRAPH_FAILED);
    RunRoceCombineCase(false, 0, 8, MIN_ASSIST_INFO_SIZE, 256, ge::GRAPH_FAILED);
}

TEST(MoeDistributeCombineV3RoceTiling, RejectsInvalidTopologyAndTruncatedAssist)
{
    RunRoceCombineCase(true, 0, 3, MIN_ASSIST_INFO_SIZE, 256, ge::GRAPH_FAILED);
    RunRoceCombineCase(true, 0, 4, MIN_ASSIST_INFO_SIZE - 1, 256, ge::GRAPH_FAILED);
}

TEST(MoeDistributeCombineV3RoceTiling, OnlyPreReduceRequiresExpandedWeights)
{
    RunRoceCombineCase(true, 0, 4, MIN_ASSIST_INFO_SIZE, 0, ge::GRAPH_FAILED);
    RunRoceCombineCase(false, 0, 4, MIN_ASSIST_INFO_SIZE, 0, ge::GRAPH_SUCCESS);
}
} // namespace
