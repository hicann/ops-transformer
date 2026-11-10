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
#include "../../../../../common/tests/ut/formal_error_assertions.h"
#include "mc2_tiling_case_executor.h"

using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

namespace MoeDistributeDispatchUT {

static constexpr uint64_t kSkipTilingKeyCheck = UINT64_MAX;
static const gert::TilingContextPara::TensorDescription kTdDefault = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

class MoeDistributeDispatchArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchArch22TilingTest TearDown" << std::endl;
    }
};

// normal: share rank
TEST_F(MoeDistributeDispatchArch22TilingTest, Test0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 512UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// normal: share rank
TEST_F(MoeDistributeDispatchArch22TilingTest, Test1)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
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
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test2)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test3)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test4)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test5)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
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
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test6)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 512UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test7)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test8)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test9)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16 * 8}, {16 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 514UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, Test10)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家 分层
// 可选传入默认值   除了quant_mode = 0
// 只改变环境变量HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE
TEST_F(MoeDistributeDispatchArch22TilingTest, A2Quant0Layered)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8 * 8 * 32}, {8 * 8 * 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeDispatchArch22TilingTest, A2Quant0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8 * 8 * 32}, {8 * 8 * 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
// A2 test
// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了quant_mode = 0
TEST_F(MoeDistributeDispatchArch22TilingTest, A2GlobalBs)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8 * 8 * 32}, {8 * 8 * 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了 quant_mode = 2 global_bs = 512  正常
TEST_F(MoeDistributeDispatchArch22TilingTest, A2ShapeAndEpRankId)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7160}, {8, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8 * 8 * 32}, {8 * 8 * 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了ep_rank_id = 33  quant_mode = 2 同时x的shape{16,7160} 异常
TEST_F(MoeDistributeDispatchArch22TilingTest, A2MoeExpertNum)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8 * 8 * 32}, {8 * 8 * 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
//  可选传入默认值   除了 quant_mode = 2 同时moe_expert_num = 257  异常
TEST_F(MoeDistributeDispatchArch22TilingTest, EpWorldSize384)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(384)},
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
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, EpWorldSize72)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: global_bs non-zero, covers CheckBatchAttrs globalBs!=0 branch (A3/A5 base tiling)
TEST_F(MoeDistributeDispatchArch22TilingTest, GlobalBsNonZero)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// Error: K > moeExpertNum
TEST_F(MoeDistributeDispatchArch22TilingTest, KGreaterThanMoeExpertNum)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: x dim0 != expertIds dim0
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidXExpertIdsDim0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 7}, {16, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expert_scales shape mismatch with expert_ids
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertScalesShape)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{7, 7}, {7, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expert_scales must be 2D
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertScalesDimNum)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{56}, {56}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expert_scales dtype must be FLOAT
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertScalesDtype)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: scales dim0 != moeExpertNum when sharedExpertRankNum == 0
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidScalesDim0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7, 7168}, {7, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expandIdx dim0 != bs * k
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpandIdxDim0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{55}, {55}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: shared expert expertTokenNums dim0 must be 1
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertTokenNumsShared)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: moe expert expertTokenNums dim0 must equal localMoeExpertNum
TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertTokenNumsMoe)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: HCCL window size too small
TEST_F(MoeDistributeDispatchArch22TilingTest, CheckWinSizeTooSmall)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}, {"cclBufferSize", 1024UL}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Success on A5 (DAV_3510 path through base tiling)
TEST_F(MoeDistributeDispatchArch22TilingTest, A5GlobalBsNonZero)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "3510", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

TEST_F(MoeDistributeDispatchArch22TilingTest, A2EpWorldSizeTopologyMismatchDoesNotBlockTiling)
{
    struct RankSizeCase {
        int64_t epWorldSize;
        uint64_t actualRankSize;
        ge::graphStatus expected;
    };
    const RankSizeCase cases[] = {
        {8, 8, ge::GRAPH_SUCCESS},   {8, 2, ge::GRAPH_SUCCESS}, {8, 16, ge::GRAPH_SUCCESS}, {16, 8, ge::GRAPH_SUCCESS},
        {16, 16, ge::GRAPH_SUCCESS}, {8, 0, ge::GRAPH_SUCCESS}, {8, 1, ge::GRAPH_SUCCESS},
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(testing::Message() << "epWorldSize=" << testCase.epWorldSize
                                        << ", actualRankSize=" << testCase.actualRankSize);
        struct MoeDistributeDispatchCompileInfo {
        } compileInfo;
        const int64_t capacity = 8 * testCase.epWorldSize;
        const int64_t epRecvCount = testCase.epWorldSize + capacity * 2 * testCase.epWorldSize / 8;
        gert::TilingContextPara tilingContextPara(
            "MoeDistributeDispatch",
            {
                {{{8, 512}, {8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{8, 1}, {8, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {
                {{{capacity, 512}, {capacity, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{capacity}, {capacity}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                {{{epRecvCount}, {epRecvCount}}, ge::DT_INT32, ge::FORMAT_ND},
                {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(testCase.epWorldSize)},
             {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(testCase.epWorldSize)},
             {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            &compileInfo, "Ascend910B", 48, 196608);
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", testCase.actualRankSize}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, testCase.expected, kSkipTilingKeyCheck);
    }
}

TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidSharedRankDivisibilityReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 512UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0027", "ep_world_size, shared_expert_rank_num",
                           "ep_world_size=288, shared_expert_rank_num=31",
                           "ep_world_size must be divisible by shared_expert_rank_num");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidExpertDivisibilityReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(255)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 512UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0027", "moe_expert_num, ep_world_size, shared_expert_rank_num",
                           "moe_expert_num=255, ep_world_size=288, shared_expert_rank_num=32",
                           "moe_expert_num must be divisible by ep_world_size - shared_expert_rank_num");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, InvalidTpWorldSizeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 512UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "tp_world_size", "3");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, ExpertScalesNzReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0035", "expert_scales", "FRACTAL_NZ");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, EpRecvCountsNzReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0017", "epRecvCounts", "FRACTAL_NZ", nullptr, "ND");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, InputRankReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0012", "xShape", "3");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, OutputRankReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 1, 7168}, {64, 1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0012", "expandXShape", "3");
}

TEST_F(MoeDistributeDispatchArch22TilingTest, DynamicScalesRankReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0011", "dynamicScales", "2D");
}

static gert::TilingContextPara OwnerDispatchBoundaryContext()
{
    struct MoeDistributeDispatchCompileInfo {};
    static MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            kTdDefault,
            kTdDefault,
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{56}, {56}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    return tilingContextPara;
}

TEST_F(MoeDistributeDispatchArch22TilingTest, ExpertScalesDtypePrecedesFormatAndOptionalAbsenceSucceeds)
{
    for (ge::Format format : {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ}) {
        ExcelErrorReportScope errors;
        auto context = OwnerDispatchBoundaryContext();
        context.inputTensorDesc_[4] = {{{8, 7}, {8, 7}}, ge::DT_FLOAT16, format};
        Mc2ExecuteTestCase(context, {{"rankNum", 8}}, ge::GRAPH_FAILED, UINT64_MAX);
        ExpectExcelFormalError("EZ0019", "expert_scales", "DT_FLOAT16", nullptr, nullptr,
                               {{"correct_dtype", "FLOAT"}, {"op_name", "MoeDistributeDispatch"}});
    }
    ExcelErrorReportScope errors;
    auto context = OwnerDispatchBoundaryContext();
    context.inputTensorDesc_[4] = kTdDefault;
    context.outputTensorDesc_.erase(context.outputTensorDesc_.begin() + 6, context.outputTensorDesc_.end());
    Mc2ExecuteTestCase(context, {{"rankNum", 8}}, ge::GRAPH_SUCCESS, UINT64_MAX);
    EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
}
TEST_F(MoeDistributeDispatchArch22TilingTest, LocalExpertOneBoundaryPreservesPositiveInvariant)
{
    for (int64_t experts : {0LL, 7LL, 8LL}) {
        SCOPED_TRACE(experts);
        ExcelErrorReportScope errors;
        auto context = OwnerDispatchBoundaryContext();
        context.inputTensorDesc_[1].shape_ = gert::StorageShape{{8, 1}, {8, 1}};
        context.inputTensorDesc_[4] = kTdDefault;
        context.outputTensorDesc_[2].shape_ = gert::StorageShape{{8}, {8}};
        context.outputTensorDesc_.erase(context.outputTensorDesc_.begin() + 6, context.outputTensorDesc_.end());
        context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(experts);
        context.attrs_[8].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(1);
        context.attrs_[9].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(0);
        Mc2ExecuteTestCase(context, {{"rankNum", 8}}, experts == 8 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, UINT64_MAX);
        if (experts == 0)
            ExpectExcelFormalError("EZ0024", "moe_expert_num", "0", nullptr, nullptr, {{"correct_value", "(0, 512]"}});
        else if (experts == 7)
            ExpectExcelFormalError("EZ0027", "moe_expert_num, ep_world_size, shared_expert_rank_num",
                                   "moe_expert_num=7, ep_world_size=8, shared_expert_rank_num=0",
                                   "moe_expert_num must be divisible by ep_world_size - shared_expert_rank_num");
        else
            EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
    }
}
} // namespace MoeDistributeDispatchUT
