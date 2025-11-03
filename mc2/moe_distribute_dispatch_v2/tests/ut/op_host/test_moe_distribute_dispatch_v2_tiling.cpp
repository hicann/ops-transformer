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
#include <map>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace MoeDistributeDispatchV2 {
class MoeDistributeDispatchV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeDispatchV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeDispatchV2Tiling TearDown" << std::endl;
    }
};

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_0)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
        // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
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
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_1)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7160}, {16, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7160}, {576, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
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
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_2)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7160}, {16, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7160}, {576, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_3)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7160}, {16, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7160}, {576, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_4)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7168}, {16, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_5)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7168}, {16, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_6)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{64, 7168}, {64, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_7)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
    &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_8)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_9)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{16, 7160}, {16, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7160}, {576, 7160}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_10)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
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
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_ep_world_size_384)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(384)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
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
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_ep_world_size_72)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_x_active_mask_2dims)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_BOOL, ge::FORMAT_ND},
            {{},  ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_elastic_info)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{32, 7168}, {32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}},  ge::DT_BOOL, ge::FORMAT_ND},
            {{},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{148}, {148}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{576, 7168}, {576, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_zeroComputeExpertNum)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{64, 7168}, {64, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_zeroComputeExpertNum_invalid)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{64, 7168}, {64, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0xffffffff)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara);
}

// =======================================A2===========================================

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_empty)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_hierarchy)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{},  ge::DT_BOOL, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("hierarchy")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2100001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_error)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("error")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);

    ExecuteTestCase(tilingContextPara);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_empty_with_env)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{},  ge::DT_INT32, ge::FORMAT_ND},
            {{},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh_with_env)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


// normal: share rank
TEST_F(MoeDistributeDispatchV2Tiling, moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh_zeroComputeExpert_not_zero)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        {
            // input info
            {{{8, 7168}, {8, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}},  ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{8*8*32, 7168}, {8*8*32, 7168}},  ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}},  ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}},  ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}},  ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}},  ge::DT_INT32, ge::FORMAT_ND},
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
        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);

    uint64_t expectTilingKey = 2000001000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

}  // moe_distribute_dispatch_v2_ut