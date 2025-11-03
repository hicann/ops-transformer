/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace MoeDistributeCombineAddRmsNormNameSpace{
struct TestParam {
    string test_name{};
    std::vector<std::pair<string, string>> tiling_params_str_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_dTypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    int64_t A{64};
    int64_t BSK{192};
    int64_t BS{8};
    int64_t K{8};
    int64_t H{7168};
    int64_t ep_world_size{8};
    int64_t ep_rank_id{0};
    int64_t moe_expert_num{8};
    int64_t tp_world_size{1};
    int64_t tp_rank_id{0};
    int64_t expert_shard_type{0};
    int64_t shared_expert_num{0};
    int64_t shared_expert_rank_num{0};
    int64_t global_bs{0};
    int64_t out_dtype{0};
    int64_t comm_quant_mode{0};
    int64_t group_list_type{0};
    float norm_eps{1e-6};
    std::string comm_alg{""};
    std::string group_ep{"group_ep"};
    std::string group_tp{"group_tp"};
};

class MoeDistributeCombineAddRmsNormTilingTest : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormTilingTest TearDown" << std::endl;
    }
};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
    tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }}};

TEST_P(MoeDistributeCombineAddRmsNormTilingTest, common_test)
{
    auto test_param = GetParam();
    auto tiling_params = TilingParams{};

    for (auto& kv : test_param.tiling_params_str_pair) {
        if (tiling_params_str_handlers.count(kv.first) != 0) {
            tiling_params_str_handlers[kv.first](tiling_params, kv.second);
        }
    }

    struct MoeDistributeCombineAddRmsNormInfo {};
    MoeDistributeCombineAddRmsNormInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineAddRmsNorm",
        {
            {{{tiling_params.A, tiling_params.H}, {tiling_params.A, tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.K}, {tiling_params.BS, tiling_params.K}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tiling_params.A * 128}, {tiling_params.A * 128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tiling_params.ep_world_size}, {tiling_params.ep_world_size}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.K}, {tiling_params.BS, tiling_params.K}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tiling_params.BS, 1, tiling_params.H}, {tiling_params.BS, 1, tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tiling_params.H}, {tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tiling_params.tp_world_size}, {tiling_params.tp_world_size}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tiling_params.BS}, {tiling_params.BS}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.H}, {tiling_params.BS, tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{tiling_params.BS, 1, tiling_params.H}, {tiling_params.BS, 1, tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tiling_params.BS, 1, 1}, {tiling_params.BS, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tiling_params.BS, 1, tiling_params.H}, {tiling_params.BS, 1, tiling_params.H}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group_ep)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group_tp)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.group_list_type)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.comm_alg)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(tiling_params.norm_eps)},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93",
        20,
        196608);
    if(test_param.status == ge::GRAPH_FAILED){
        ExecuteTestCase(tilingContextPara);
    }
    else {
        uint64_t expectTilingKey = 10000UL;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

static TestParam test_params[] = {
    {"Test_sample", {}, {}, ge::GRAPH_SUCCESS}
};

INSTANTIATE_TEST_SUITE_P(MoeDistributeCombineAddRmsNormTilingTest, MoeDistributeCombineAddRmsNormTilingTest,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<MoeDistributeCombineAddRmsNormTilingTest::ParamType>& info) {
                             return info.param.test_name;
                         });

} // MoeDistributeCombineAddRmsNormNameSpace
