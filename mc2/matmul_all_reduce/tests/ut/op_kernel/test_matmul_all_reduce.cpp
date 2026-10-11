/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_matmul_all_reduce.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include "matmul_all_reduce_tiling_def.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/matmul_all_reduce.cpp"
using namespace std;
using namespace Mc2Tiling;

extern uint8_t* g_hcclContextReserved[2];
struct HcclCombinOpParams {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

class matmul_all_reduce_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclCombinOpParams);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        cout << "matmul_all_reduce_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        cout << "matmul_all_reduce_test TearDown\n" << endl;
    }
};

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_no_bias)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData* tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmul_all_reduce_wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                        GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                        GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        matmul_all_reduce<ASCEND_910B, MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY, MATMUL_ALLREDUCE_EMPTY_INPUT_F,
                          MATMUL_ALLREDUCE_INT8_COMM_F, 0, 0, SET_NOT_USE_FM_MM_TPL_TILING,
                          SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(260);
    ICPU_RUN_KF(matmul_all_reduce_wrapper, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_bias_cube)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData* tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.biasLen = 1536 * sizeof(float);
    tilingData->matmulTiling.shareUbSize = 192 * 1024;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmul_all_reduce_wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                        GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                        GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        matmul_all_reduce<ASCEND_910B, MATMUL_ALLREDUCE_MM_TYPE_FP_MM, MATMUL_ALLREDUCE_EMPTY_INPUT_F,
                          MATMUL_ALLREDUCE_INT8_COMM_F, 0, 0, MAT_MUL_V3_MIXND2NZ_TRUE, SET_NOT_USE_QUANT_MM_TPL_TILING,
                          SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(256);
    ICPU_RUN_KF(matmul_all_reduce_wrapper, 20, aGM, bGM, biasGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_no_bias_l2cache_cube)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData* tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 8;
    tilingData->param.rankDim = 1;
    tilingData->param.tileCnt = 1;
    Mc2Tiling::Mc2L2cacheTilePara tileL2cacheTiling;
    tilingData->tileL2cacheTiling.mTileCntL2 = 1;
    tilingData->tileL2cacheTiling.nTileCntL2 = 2;
    tilingData->tileL2cacheTiling.mTileBlock = 1;
    tilingData->tileL2cacheTiling.nTileBlock = 16;
    tilingData->matmulTiling.M = 128;
    tilingData->matmulTiling.N = 8192;
    tilingData->matmulTiling.Ka = 1024;
    tilingData->matmulTiling.Kb = 1024;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 256;
    tilingData->matmulTiling.singleCoreK = 1024;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 64;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.usedCoreNum = 8;
    tilingData->matmulTiling.shareUbSize = 192 * 1024;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(8192 * 1024 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(128 * 8192 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmul_all_reduce_wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                        GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                        GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        matmul_all_reduce<ASCEND_910B, MATMUL_ALLREDUCE_MM_TYPE_FP_MM, MATMUL_ALLREDUCE_EMPTY_INPUT_F,
                          MATMUL_ALLREDUCE_INT8_COMM_F, 0, 0, MAT_MUL_V3_MIXND2NZ_FALSE,
                          SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(matmul_all_reduce_wrapper, 8, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_empty_k)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData* tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailCnt = 0;
    tilingData->param.rankDim = 1;
    tilingData->param.tileCnt = 1;
    tilingData->matmulTiling.M = 128;
    tilingData->matmulTiling.N = 1024;
    tilingData->matmulTiling.Ka = 0;
    tilingData->matmulTiling.Kb = 0;
    tilingData->matmulTiling.singleCoreM = 0;
    tilingData->matmulTiling.singleCoreN = 0;
    tilingData->matmulTiling.singleCoreK = 0;
    tilingData->matmulTiling.baseM = 0;
    tilingData->matmulTiling.baseN = 0;
    tilingData->matmulTiling.baseK = 0;
    tilingData->matmulTiling.stepKa = 0;
    tilingData->matmulTiling.stepKb = 0;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.depthA1 = 0;
    tilingData->matmulTiling.depthB1 = 0;
    tilingData->matmulTiling.stepM = 0;
    tilingData->matmulTiling.stepN = 0;
    tilingData->matmulTiling.usedCoreNum = 1;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(128 * 0 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(0 * 1024 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmul_all_reduce_wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                        GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                        GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        matmul_all_reduce<ASCEND_910B, MATMUL_ALLREDUCE_MM_TYPE_FP_MM, MATMUL_ALLREDUCE_EMPTY_INPUT_T,
                          MATMUL_ALLREDUCE_INT8_COMM_F, 0, 0, MAT_MUL_V3_MIXND2NZ_FALSE,
                          SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(16);
    ICPU_RUN_KF(matmul_all_reduce_wrapper, 8, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}
