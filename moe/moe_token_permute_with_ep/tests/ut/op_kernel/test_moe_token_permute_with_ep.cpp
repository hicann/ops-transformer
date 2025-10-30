/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_token_permute_with_ep_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_permute_with_ep(
    GM_ADDR tokens, GM_ADDR indices, GM_ADDR probs, GM_ADDR permuteTokens, GM_ADDR sortedIndices, GM_ADDR permuteProbs,
    GM_ADDR workspace, GM_ADDR tiling);
class moe_token_permute_with_ep_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_token_permute_with_ep_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_token_permute_with_ep_test TearDown\n" << endl;
    }
};

TEST_F(moe_token_permute_with_ep_test, test_token_permute_0)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_1)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 2;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_2)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 3;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_3)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 4;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_4)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 5;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_5)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 6;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_6)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 7;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(moe_token_permute_with_ep_test, test_token_permute_7)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    size_t active_rows = 6;
    uint64_t tilingKey = 8;
    uint32_t blockDim = 1;

    size_t out_row = num_rows > active_rows ? active_rows : num_rows;

    size_t x_FileSize = num_rows * cols * sizeof(half);
    size_t rowIdx_FileSize = num_rows * k * sizeof(int64_t);
    size_t probs_FileSize = num_rows * k * sizeof(half);
    size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
    size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedProbs_FileSize = out_row * k * sizeof(float);
    size_t workspace_FileSize =
        num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
    size_t tilingDataSize = sizeof(MoeTokenPermuteWithEpTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probs_FileSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
    uint8_t* expandedProbs = (uint8_t*)AscendC::GmAlloc(expandedProbs_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        moe_token_permute_with_ep, blockDim, x, rowIdx, probs, expandedX, expandedExpertIdx, expandedProbs, workspace,
        tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)rowIdx);
    AscendC::GmFree((void*)probs);
    AscendC::GmFree((void*)expandedX);
    AscendC::GmFree((void*)expandedExpertIdx);
    AscendC::GmFree((void*)expandedProbs);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}