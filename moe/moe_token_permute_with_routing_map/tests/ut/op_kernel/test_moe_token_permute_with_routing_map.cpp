/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_token_permute_with_routing_map_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map(GM_ADDR tokens, GM_ADDR indices, GM_ADDR prob,
                                                                         GM_ADDR permuteTokens, GM_ADDR permuteProb,
                                                                         GM_ADDR sortedIndices, GM_ADDR workspace,
                                                                         GM_ADDR tiling);
class moe_token_permute_with_routing_map_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "moe_token_permute_with_routing_map_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_token_permute_with_routing_map_test TearDown\n" << endl;
    }
};

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_0)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);
    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_1)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 2;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_2)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 3;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_3)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 4;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_4)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 5;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_5)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 6;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_6)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 7;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_routing_map_test, test_token_permute_7)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = num_rows * k;
    uint64_t tilingKey = 9;
    uint32_t blockDim = 1;

    size_t out_row = active_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(bool);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t prob_FileSize = num_rows * k * sizeof(half);
    size_t permuteProb_FileSize = out_row * sizeof(half);
    uint8_t* prob = (uint8_t*)AscendC::GmAlloc(prob_FileSize);
    uint8_t* permuteProb = (uint8_t*)AscendC::GmAlloc(permuteProb_FileSize);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_token_permute_with_routing_map, blockDim, x, rowIdx, prob, expandedX, permuteProb,
                expandedExpertIdx, workspace, tiling);

    AscendC::GmFree((void*)prob);
    AscendC::GmFree((void*)permuteProb);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}
