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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace MoeDistributeDispatchUT {

class MoeDistributeCombineTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeCombineTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeCombineTiling TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_0) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7*8}, {7*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);;
    uint64_t expectTilingKey = 1000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_1) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    int64_t ep_world_size = 288;
    int64_t tp_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_2) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    int64_t ep_world_size = 288;
    int64_t tp_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 1024;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_3) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    int64_t ep_world_size = 288;
    int64_t tp_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 31;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_layered) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_global_bs) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 0;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 512;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_shape) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 0;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7160}, {8*8*32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_ep_rankId) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 33;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 0;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_moe_expert_num) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 0;
    int64_t moe_expert_num = 257;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}

TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_ep_world_size_384) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    int64_t ep_world_size = 384;
    int64_t tp_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_ep_world_size_72) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    int64_t ep_world_size = 72;
    int64_t tp_world_size = 2;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 18;
    int64_t moe_expert_num = 216;
    int64_t global_bs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_int8_quant) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    uint64_t expectTilingKey = 2000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeCombineTiling, moe_distribute_combine_test_tiling_A2_invalid_comm_quant_mode) {
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string ep_group("ep_group");
    std::string tp_group("");
    int64_t ep_world_size = 32;
    int64_t tp_world_size = 0;
    int64_t ep_rank_id = 0;
    int64_t tp_rank_id = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 3;
    int64_t group_list_type = 0;
    int64_t shared_expert_rank_num = 32;
    int64_t moe_expert_num = 256;
    int64_t global_bs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombine",
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(ep_group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tp_group)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_list_type)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
        ExecuteTestCase(tilingContextPara);
}

} //moe_distribute_combine_ut