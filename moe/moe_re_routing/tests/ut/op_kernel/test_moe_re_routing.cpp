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
 * \file test_moe_re_routing.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include <cstdint>
#include "data_utils.h"
#include "test_moe_re_routing.h"

using namespace std;

extern "C" __global__ __aicore__ void moe_re_routing(GM_ADDR tokens, GM_ADDR expertTokensNumPerRank,
    GM_ADDR perTokenScales, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
    GM_ADDR expertTokenNum, GM_ADDR workspace, GM_ADDR tiling);

class moe_re_routing_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "moe_re_routing_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_re_routing_test TearDown\n" << endl;
    }
};

TEST_F(moe_re_routing_test, test_case_0001)
{
    size_t tokensSize = 16384 * 7168 * sizeof(int8_t);
    size_t expertTokensNumPerRankSize = 16 * 16 * sizeof(int32_t);
    size_t perTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokensSize = 16384 * 7168 * sizeof(int8_t);
    size_t permutePerTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokenIdxSize = 16384 * sizeof(int32_t);
    size_t expertTokenNumSize = 16 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(MoeReRoutingTilingData);
    uint32_t blockDim = 48;

    uint8_t *tokens = (uint8_t *)AscendC::GmAlloc(tokensSize);
    uint8_t *expertTokenNumPerRank = (uint8_t *)AscendC::GmAlloc(expertTokensNumPerRankSize);
    uint8_t *perTokenScales = (uint8_t *)AscendC::GmAlloc(perTokenScalesSize);
    uint8_t *permuteTokens = (uint8_t *)AscendC::GmAlloc(permuteTokensSize);
    uint8_t *permutePerTokenScales = (uint8_t *)AscendC::GmAlloc(permutePerTokenScalesSize);
    uint8_t *permuteTokenIdx = (uint8_t *)AscendC::GmAlloc(permuteTokenIdxSize);
    uint8_t *expertTokenNum = (uint8_t *)AscendC::GmAlloc(expertTokenNumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MoeReRoutingTilingData *tilingDatafromBin = reinterpret_cast<MoeReRoutingTilingData *>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->ubFactor = 12;
    tilingDatafromBin->tokensNum = 16384;
    tilingDatafromBin->tokensSize = 7168;
    tilingDatafromBin->rankNum = 16;
    tilingDatafromBin->expertNumPerRank = 16;
    tilingDatafromBin->hasScale = 1;

    ICPU_SET_TILING_KEY(100000);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(tokens);
    AscendC::GmFree(expertTokenNumPerRank);
    AscendC::GmFree(perTokenScales);
    AscendC::GmFree(permuteTokens);
    AscendC::GmFree(permutePerTokenScales);
    AscendC::GmFree(permuteTokenIdx);
    AscendC::GmFree(expertTokenNum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_re_routing_test, test_case_0002)
{
    size_t tokensSize = 16384 * 7168 * sizeof(half);
    size_t expertTokensNumPerRankSize = 16 * 16 * sizeof(int32_t);
    size_t perTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokensSize = 16384 * 7168 * sizeof(half);
    size_t permutePerTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokenIdxSize = 16384 * sizeof(int32_t);
    size_t expertTokenNumSize = 16 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(MoeReRoutingTilingData);
    uint32_t blockDim = 48;

    uint8_t *tokens = (uint8_t *)AscendC::GmAlloc(tokensSize);
    uint8_t *expertTokenNumPerRank = (uint8_t *)AscendC::GmAlloc(expertTokensNumPerRankSize);
    uint8_t *perTokenScales = (uint8_t *)AscendC::GmAlloc(perTokenScalesSize);
    uint8_t *permuteTokens = (uint8_t *)AscendC::GmAlloc(permuteTokensSize);
    uint8_t *permutePerTokenScales = (uint8_t *)AscendC::GmAlloc(permutePerTokenScalesSize);
    uint8_t *permuteTokenIdx = (uint8_t *)AscendC::GmAlloc(permuteTokenIdxSize);
    uint8_t *expertTokenNum = (uint8_t *)AscendC::GmAlloc(expertTokenNumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MoeReRoutingTilingData *tilingDatafromBin = reinterpret_cast<MoeReRoutingTilingData *>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->ubFactor = 5;
    tilingDatafromBin->tokensNum = 16384;
    tilingDatafromBin->tokensSize = 7168;
    tilingDatafromBin->rankNum = 16;
    tilingDatafromBin->expertNumPerRank = 16;
    tilingDatafromBin->hasScale = 1;

    ICPU_SET_TILING_KEY(100100);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(100200);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(tokens);
    AscendC::GmFree(expertTokenNumPerRank);
    AscendC::GmFree(perTokenScales);
    AscendC::GmFree(permuteTokens);
    AscendC::GmFree(permutePerTokenScales);
    AscendC::GmFree(permuteTokenIdx);
    AscendC::GmFree(expertTokenNum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_re_routing_test, test_case_0003)
{
    size_t tokensSize = 16384 * 7168 * sizeof(int8_t);
    size_t expertTokensNumPerRankSize = 16 * 16 * sizeof(int64_t);
    size_t perTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokensSize = 16384 * 7168 * sizeof(int8_t);
    size_t permutePerTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokenIdxSize = 16384 * sizeof(int32_t);
    size_t expertTokenNumSize = 16 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(MoeReRoutingTilingData);
    uint32_t blockDim = 48;

    uint8_t *tokens = (uint8_t *)AscendC::GmAlloc(tokensSize);
    uint8_t *expertTokenNumPerRank = (uint8_t *)AscendC::GmAlloc(expertTokensNumPerRankSize);
    uint8_t *perTokenScales = (uint8_t *)AscendC::GmAlloc(perTokenScalesSize);
    uint8_t *permuteTokens = (uint8_t *)AscendC::GmAlloc(permuteTokensSize);
    uint8_t *permutePerTokenScales = (uint8_t *)AscendC::GmAlloc(permutePerTokenScalesSize);
    uint8_t *permuteTokenIdx = (uint8_t *)AscendC::GmAlloc(permuteTokenIdxSize);
    uint8_t *expertTokenNum = (uint8_t *)AscendC::GmAlloc(expertTokenNumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MoeReRoutingTilingData *tilingDatafromBin = reinterpret_cast<MoeReRoutingTilingData *>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->ubFactor = 12;
    tilingDatafromBin->tokensNum = 16384;
    tilingDatafromBin->tokensSize = 7168;
    tilingDatafromBin->rankNum = 16;
    tilingDatafromBin->expertNumPerRank = 16;
    tilingDatafromBin->hasScale = 1;

    ICPU_SET_TILING_KEY(100010);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(tokens);
    AscendC::GmFree(expertTokenNumPerRank);
    AscendC::GmFree(perTokenScales);
    AscendC::GmFree(permuteTokens);
    AscendC::GmFree(permutePerTokenScales);
    AscendC::GmFree(permuteTokenIdx);
    AscendC::GmFree(expertTokenNum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_re_routing_test, test_case_0004)
{
    size_t tokensSize = 16384 * 7168 * sizeof(half);
    size_t expertTokensNumPerRankSize = 16 * 16 * sizeof(int64_t);
    size_t perTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokensSize = 16384 * 7168 * sizeof(half);
    size_t permutePerTokenScalesSize = 16384 * sizeof(float);
    size_t permuteTokenIdxSize = 16384 * sizeof(int32_t);
    size_t expertTokenNumSize = 16 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(MoeReRoutingTilingData);
    uint32_t blockDim = 48;

    uint8_t *tokens = (uint8_t *)AscendC::GmAlloc(tokensSize);
    uint8_t *expertTokenNumPerRank = (uint8_t *)AscendC::GmAlloc(expertTokensNumPerRankSize);
    uint8_t *perTokenScales = (uint8_t *)AscendC::GmAlloc(perTokenScalesSize);
    uint8_t *permuteTokens = (uint8_t *)AscendC::GmAlloc(permuteTokensSize);
    uint8_t *permutePerTokenScales = (uint8_t *)AscendC::GmAlloc(permutePerTokenScalesSize);
    uint8_t *permuteTokenIdx = (uint8_t *)AscendC::GmAlloc(permuteTokenIdxSize);
    uint8_t *expertTokenNum = (uint8_t *)AscendC::GmAlloc(expertTokenNumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MoeReRoutingTilingData *tilingDatafromBin = reinterpret_cast<MoeReRoutingTilingData *>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->ubFactor = 5;
    tilingDatafromBin->tokensNum = 16384;
    tilingDatafromBin->tokensSize = 7168;
    tilingDatafromBin->rankNum = 16;
    tilingDatafromBin->expertNumPerRank = 16;
    tilingDatafromBin->hasScale = 1;

    ICPU_SET_TILING_KEY(100110);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(100210);
    ICPU_RUN_KF(moe_re_routing,
        blockDim,
        tokens,
        expertTokenNumPerRank,
        perTokenScales,
        permuteTokens,
        permutePerTokenScales,
        permuteTokenIdx,
        expertTokenNum,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(tokens);
    AscendC::GmFree(expertTokenNumPerRank);
    AscendC::GmFree(perTokenScales);
    AscendC::GmFree(permuteTokens);
    AscendC::GmFree(permutePerTokenScales);
    AscendC::GmFree(permuteTokenIdx);
    AscendC::GmFree(expertTokenNum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}