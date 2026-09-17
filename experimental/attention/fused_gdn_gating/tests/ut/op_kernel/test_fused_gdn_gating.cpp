/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_gdn_gating.cpp
 * \brief Kernel unit tests for fused_gdn_gating.
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "fused_gdn_gating_tiling_data.h"

using namespace std;
using FusedGdnGating::FusedGdnGatingTilingData;

extern "C" __global__ __aicore__ void fused_gdn_gating(GM_ADDR a_log, GM_ADDR a, GM_ADDR b, GM_ADDR dt_bias, GM_ADDR g,
                                                       GM_ADDR beta_output, GM_ADDR workspace, GM_ADDR tiling_gm);

template <typename T>
T *GmAllocWrapper(size_t size)
{
    T *ptr = reinterpret_cast<T *>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

void InitTilingData(FusedGdnGatingTilingData *td, uint32_t numHeads, uint32_t numBatches, float beta, float threshold)
{
    td->numHeads = numHeads;
    td->beta = beta;
    td->numBatches = numBatches;
    td->rowsPerIter = 1;
    td->useBulkDma = 0;
    td->usedCoreNum = 1;
    td->alignedLength = 0;
    td->tailLength = 0;
    td->tileRows = 0;
    td->inv_beta = (std::abs(beta) < 1e-6f) ? std::numeric_limits<float>::infinity() : (1.0f / beta);
    td->threshold = threshold;
}

void InitInputData(uint8_t *aLogGm, size_t shapeALog, uint8_t *aGm, size_t shapeA, uint8_t *bGm, size_t shapeB,
                   uint8_t *dtBiasGm, size_t shapeDtBias, uint8_t *gGm, size_t shapeG, uint8_t *betaOutputGm,
                   size_t shapeBetaOutput)
{
    memset(aLogGm, 0, shapeALog);
    memset(aGm, 0, shapeA);
    memset(bGm, 0, shapeB);
    memset(dtBiasGm, 0, shapeDtBias);
    memset(gGm, 0, shapeG);
    memset(betaOutputGm, 0, shapeBetaOutput);
}

struct FggTestParams {
    uint32_t tilingKey;
};

class FusedGdnGatingKernelTest : public testing::TestWithParam<FggTestParams> {
protected:
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    size_t shapeALog = numHeads * sizeof(float);
    size_t shapeA = batch * numHeads * sizeof(bfloat16_t);
    size_t shapeB = batch * numHeads * sizeof(bfloat16_t);
    size_t shapeDtBias = numHeads * sizeof(float);
    size_t shapeG = 1 * batch * numHeads * sizeof(float);
    size_t shapeBetaOutput = 1 * batch * numHeads * sizeof(bfloat16_t);
    size_t allWorkspaceSize = 196608;
    size_t tilingSize = sizeof(FusedGdnGatingTilingData);

    uint8_t *aLogGm = nullptr;
    uint8_t *aGm = nullptr;
    uint8_t *bGm = nullptr;
    uint8_t *dtBiasGm = nullptr;
    uint8_t *gGm = nullptr;
    uint8_t *betaOutputGm = nullptr;
    uint8_t *workspace = nullptr;
    uint8_t *tiling = nullptr;

    void SetUp() override
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        aLogGm = GmAllocWrapper<uint8_t>(shapeALog);
        aGm = GmAllocWrapper<uint8_t>(shapeA);
        bGm = GmAllocWrapper<uint8_t>(shapeB);
        dtBiasGm = GmAllocWrapper<uint8_t>(shapeDtBias);
        gGm = GmAllocWrapper<uint8_t>(shapeG);
        betaOutputGm = GmAllocWrapper<uint8_t>(shapeBetaOutput);
        workspace = GmAllocWrapper<uint8_t>(allWorkspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        InitInputData(aLogGm, shapeALog, aGm, shapeA, bGm, shapeB, dtBiasGm, shapeDtBias, gGm, shapeG, betaOutputGm,
                      shapeBetaOutput);

        auto params = GetParam();
        FusedGdnGatingTilingData *td = reinterpret_cast<FusedGdnGatingTilingData *>(tiling);
        InitTilingData(td, numHeads, batch, 1.0f, 20.0f);
    }

    void TearDown() override
    {
        AscendC::GmFree(aLogGm);
        AscendC::GmFree(aGm);
        AscendC::GmFree(bGm);
        AscendC::GmFree(dtBiasGm);
        AscendC::GmFree(gGm);
        AscendC::GmFree(betaOutputGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

INSTANTIATE_TEST_SUITE_P(GeneralTests, FusedGdnGatingKernelTest, testing::Values(FggTestParams{1}));

TEST_P(FusedGdnGatingKernelTest, RunTest)
{
    auto params = GetParam();
    std::cout << "test config: tilingKey=" << params.tilingKey << std::endl;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(params.tilingKey);
    ICPU_RUN_KF(fused_gdn_gating, blockDim, aLogGm, aGm, bGm, dtBiasGm, gGm, betaOutputGm, workspace, tiling);
}

#if __CCE_AICORE__ == 200

class FusedGdnGating310PTest : public testing::Test {
protected:
    uint32_t batch = 4;
    uint32_t numHeads = 8;
    uint32_t totalElements = batch * numHeads;

    size_t shapeALog = numHeads * sizeof(half);
    size_t shapeA = totalElements * sizeof(half);
    size_t shapeB = totalElements * sizeof(half);
    size_t shapeDtBias = numHeads * sizeof(half);
    size_t shapeG = totalElements * sizeof(float);
    size_t shapeBetaOutput = totalElements * sizeof(half);
    size_t allWorkspaceSize = 196608;
    size_t tilingSize = sizeof(FusedGdnGatingTilingData);

    uint8_t *aLogGm = nullptr;
    uint8_t *aGm = nullptr;
    uint8_t *bGm = nullptr;
    uint8_t *dtBiasGm = nullptr;
    uint8_t *gGm = nullptr;
    uint8_t *betaOutputGm = nullptr;
    uint8_t *workspace = nullptr;
    uint8_t *tiling = nullptr;

    void SetUp() override
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        aLogGm = GmAllocWrapper<uint8_t>(shapeALog);
        aGm = GmAllocWrapper<uint8_t>(shapeA);
        bGm = GmAllocWrapper<uint8_t>(shapeB);
        dtBiasGm = GmAllocWrapper<uint8_t>(shapeDtBias);
        gGm = GmAllocWrapper<uint8_t>(shapeG);
        betaOutputGm = GmAllocWrapper<uint8_t>(shapeBetaOutput);
        workspace = GmAllocWrapper<uint8_t>(allWorkspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        InitInputData(aLogGm, shapeALog, aGm, shapeA, bGm, shapeB, dtBiasGm, shapeDtBias, gGm, shapeG, betaOutputGm,
                      shapeBetaOutput);

        FusedGdnGatingTilingData *td = reinterpret_cast<FusedGdnGatingTilingData *>(tiling);
        td->numHeads = numHeads;
        td->beta = 1.0f;
        td->numBatches = batch;
        td->rowsPerIter = 0;
        td->useBulkDma = 0;
        td->usedCoreNum = 1;
        td->alignedLength = totalElements;
        td->tailLength = totalElements;
        td->tileRows = totalElements;
        td->inv_beta = 1.0f;
        td->threshold = 20.0f;
    }

    void TearDown() override
    {
        AscendC::GmFree(aLogGm);
        AscendC::GmFree(aGm);
        AscendC::GmFree(bGm);
        AscendC::GmFree(dtBiasGm);
        AscendC::GmFree(gGm);
        AscendC::GmFree(betaOutputGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

TEST_F(FusedGdnGating310PTest, RunTest310P)
{
    std::cout << "test config: 310P tilingKey=200000" << std::endl;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(200000);
    ICPU_RUN_KF(fused_gdn_gating, blockDim, aLogGm, aGm, bGm, dtBiasGm, gGm, betaOutputGm, workspace, tiling);
}

#endif
