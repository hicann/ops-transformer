/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ts_moe_init_routing_v2_tc.cpp
 * \brief MoeInitRoutingV2用例.
 */

#include "ts_moe_init_routing_v2.h"

MoeInitRoutingV2Case InitNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                    int64_t dropPadMode, int64_t countFlag, bool tokenFlag,
                                    ge::DataType optionalOutputDt, ge::graphStatus result, int64_t tilingKey)
{
    MoeInitRoutingV2Case cs;
    cs.mParam = {N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, optionalOutputDt};
    if (result == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    } else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E, int64_t dropPadMode,
                          int64_t countFlag, bool tokenFlag, ge::DataType optionalOutputDt, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeInitRoutingV2Case cs = InitNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag,
                                             optionalOutputDt, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_perf_template)
{
    InitAndRunNormalCase(8, 30, 6, 32, 0, 8, 0, 1, false, ge::DT_INT32, ge::GRAPH_SUCCESS, 20000);
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_single_core_drop)
{
    InitAndRunNormalCase(8, 30, 6, 0, 6, 8, 1, 0, true, ge::DT_INT32, ge::GRAPH_SUCCESS, 10011);
}

// x_shape dim != 2
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_x_dim_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(320, 60, 56, 1000, 0, 64, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    cs.x = Tensor("x", {cs.mParam.n}, "1", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

// expert_tokens_before_capacity_shape_flag =1 but E = 0
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_e_check)
{
    InitAndRunNormalCase(320, 3000, 56, 1000, 0, 0, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
}

// dropPadMode = 1 but C = 0
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_c_check)
{
    InitAndRunNormalCase(320, 3000, 56, 1000, 0, 64, 1, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
}

// expanded_x_shape shape != (E, C, H)
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_droppad_expanded_x_shape_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(320, 3000, 56, 0, 30, 64, 1, 0, true, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    cs.expandedX =
        Tensor("expandedX", {cs.mParam.k * cs.mParam.n, cs.mParam.h}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape second dim != C
    cs.expandedX = Tensor("expandedX", {cs.mParam.e, cs.mParam.c * 10, cs.mParam.h}, "3", cs.mParam.optionalOutputDt,
                          ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape first dim != E
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h}, "3",
                          cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape third dim != H
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h + 1}, "3",
                          cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

// expert_tokens_before_capacity_shape dirst dim != E
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_tiling_expert_tokens_before_capacity_shape_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(320, 3000, 56, 0, 30, 64, 1, 0, true, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    cs.expertTokensBeforeCapacity =
        Tensor("expertTokensBeforeCapacity", {cs.mParam.e * 2}, "1", ge::DT_INT32, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_tiling_activenum_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(320, 3000, 56, 0, 30, 64, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    // expanded_x_shape shape != (N * k, H)
    cs.expandedX = Tensor("expandedX", {64, 30, 3000}, "3", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {2939, 3000}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {300, 3001}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_negative_check)
{
    // activeNum < 0
    InitAndRunNormalCase(320, 3000, 56, -3, 0, 64, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    // C < 0
    InitAndRunNormalCase(320, 3000, 56, 0, -5, 64, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
    // E < 0
    InitAndRunNormalCase(320, 3000, 56, 0, 0, -4, 0, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
}

// N < capacity
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_capacity_check)
{
    InitAndRunNormalCase(160, 30000, 12, 0, 1000, 32, 1, 0, false, ge::DT_INT32, ge::GRAPH_FAILED, 0);
}