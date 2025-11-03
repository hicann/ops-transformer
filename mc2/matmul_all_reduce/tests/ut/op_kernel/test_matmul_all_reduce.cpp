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
 * \file test_matmul_all_reduce.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include <unistd.h>

using namespace std;

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include "matmul_all_reduce_tiling_def.h"
// #include "quant_matmul_all_reduce_tiling_def.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

extern "C" __global__ __aicore__ void matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                                        GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
                                                        GM_ADDR dequantGM, GM_ADDR pertokenGM, 
                                                        GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern uint8_t* g_hcclContextReserved[2];

struct HcclCombinOpParam {
  uint64_t WorkSpace;
  uint64_t WorkSpaceSize;
  uint32_t rankId;
  uint32_t rankDim;
};

class matmul_all_reduce_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        cout << "matmul_all_reduce_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        cout << "matmul_all_reduce_test TearDown\n" << endl;
    }
};

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_no_bias) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(10000000000000001100UL);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_bias_cube) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.biasLen = 1536 * sizeof(float);
    tilingData->matmulTiling.shareUbSize = 192 * 1024;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(10000000000000000001UL);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, biasGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_no_bias_l2cache_cube) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 8;
    tilingData->param.rankDim = 1;
    tilingData->param.tileCnt = 1;
    Mc2L2cacheTilePara tileL2cacheTiling;
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

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(8192 * 1024 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(128 * 8192 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(10000000000000000000UL);
    ICPU_RUN_KF(matmul_all_reduce, 8, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_bias) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, biasGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_no_bias_l2cache) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 8;
    tilingData->param.rankDim = 1;
    tilingData->param.tileCnt = 1;
    Mc2L2cacheTilePara tileL2cacheTiling;
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

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(8192 * 1024 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(128 * 8192 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(200001);
    ICPU_RUN_KF(matmul_all_reduce, 8, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_201111) {
    #define DTYPE_X2 int8_t
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 8;
    tilingData->param.rankDim = 1;
    tilingData->param.tileCnt = 1;
    Mc2L2cacheTilePara tileL2cacheTiling;
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

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(8192 * 1024 * sizeof(uint8_t));
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(8192 * sizeof(uint16_t));
    uint8_t *offsetGM = (uint8_t *)AscendC::GmAlloc(8192 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(128 * 8192 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(201111);
    ICPU_RUN_KF(matmul_all_reduce, 8, aGM, bGM, nullptr, nullptr, scaleGM, offsetGM, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
    #define DTYPE_X2 half
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_quant) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint8_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint8_t));
    uint8_t *dequantGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint64_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(50);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, dequantGM, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)dequantGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_quant_bf16) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint8_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint8_t));
    uint8_t *dequantGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(bfloat16_t));
    uint8_t *addGM = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(bfloat16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(bfloat16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(50);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, dequantGM, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)dequantGM);
    AscendC::GmFree((void*)addGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_debugMode_cube) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->msg.debugMode = 1;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_debugMode_aicpu) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->msg.debugMode = 4;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(matmul_all_reduce, 20, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(matmul_all_reduce_test, matmul_all_reduce_test_empty_k) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceTilingData *tilingData = reinterpret_cast<MatmulAllReduceTilingData*>(tiling);
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

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(128 * 0 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(0 * 1024 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(128 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    ICPU_SET_TILING_KEY(10000000000000000009UL);
    ICPU_RUN_KF(matmul_all_reduce, 8, aGM, bGM, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}
