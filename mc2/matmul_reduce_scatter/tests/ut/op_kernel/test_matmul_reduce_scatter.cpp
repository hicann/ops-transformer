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
#include "gtest/gtest.h"

#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "matmul_reduce_scatter_tiling_def.h"
#include "../../../../tests/ut/framework_normal/op_kernel/data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/matmul_reduce_scatter.cpp"

using namespace std;

extern uint8_t* g_hcclContextReserved[2];

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class MatmulReduceScatterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        cout << "MatmulReduceScatterTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        cout << "MatmulReduceScatterTest TearDown\n" << endl;
    }
};

TEST_F(MatmulReduceScatterTest, MatmulReduceScatterTestNoBias)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulReduceScatterTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulReduceScatterTilingData* tilingData = reinterpret_cast<MatmulReduceScatterTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmulReduceScatterWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM) {
        matmul_reduce_scatter<true, false, false>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(MatmulReduceScatterTest, MatmulReduceScatterBias)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulReduceScatterTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulReduceScatterTilingData* tilingData = reinterpret_cast<MatmulReduceScatterTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.biasLen = 1536 * sizeof(float);

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmulReduceScatterWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM) {
        matmul_reduce_scatter<true, false, false>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, biasGM, output, workspace, tiling);
    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(MatmulReduceScatterTest, MatmulReduceScatterTestComputationOnly)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulReduceScatterTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulReduceScatterTilingData* tilingData = reinterpret_cast<MatmulReduceScatterTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmulReduceScatterWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM) {
        matmul_reduce_scatter<true, false, false>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(MatmulReduceScatterTest, MatmulReduceScatterTestCommunicationOnly)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulReduceScatterTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulReduceScatterTilingData* tilingData = reinterpret_cast<MatmulReduceScatterTilingData*>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmulReduceScatterWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM) {
        matmul_reduce_scatter<true, false, false>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}

TEST_F(MatmulReduceScatterTest, MatmulReduceScatterTestNoBiasNormalization)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulReduceScatterTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulReduceScatterTilingData* tilingData = reinterpret_cast<MatmulReduceScatterTilingData*>(tiling);
    tilingData->param.tailM = 372;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.rankDim = 8;
    tilingData->param.rankM = 1012;
    tilingData->param.tileCnt = 5;
    tilingData->param.rankN = 768;
    tilingData->param.rankK = 6144;

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(1012 * 6144 * sizeof(uint16_t));
    uint8_t* bGM = (uint8_t*)AscendC::GmAlloc(768 * 6144 * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(1012 * 768 * sizeof(uint16_t));
    uint8_t* aicpuWin = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto matmulReduceScatterWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM) {
        matmul_reduce_scatter<true, false, false>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(matmulReduceScatterWrapper, 20, aGM, bGM, nullptr, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    AscendC::GmFree((void*)aicpuWin);
}
