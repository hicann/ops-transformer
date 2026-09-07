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
#include "all_gather_matmul_tiling_def.h"
#include "../../../op_kernel/all_gather_matmul.cpp"

extern uint8_t *g_hcclContextReserved[2];

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

class AllGatherMatmulTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t *)AscendC::GmAlloc(ctxSize);
        std::cout << "AllGatherMatmulTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        AscendC::GmFree((void *)g_hcclContextReserved[0]);
        std::cout << "AllGatherMatmulTest TearDown\n" << std::endl;
    }
};

TEST_F(AllGatherMatmulTest, AllGatherMatmulTestNoBias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./all_gather_matmul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    // uint8_t *gatherOutput = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, nullptr, output, nullptr, workspace, tiling);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, nullptr, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}

TEST_F(AllGatherMatmulTest, AllGatherMatmulBias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./all_gather_matmul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    // uint8_t *gatherOutput = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, biasGM, output, nullptr, workspace, tiling);
    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, biasGM, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}

TEST_F(AllGatherMatmulTest, AllGatherMatmulTestNoBiasL2cache)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./all_gather_matmul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 128;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.rankDim = 8;
    tilingData->param.tileCnt = 2;
    tilingData->tileL2Tiling.mL2TileCnt = 1;
    tilingData->tileL2Tiling.nL2TileCnt = 2;
    tilingData->tileL2Tiling.mTileBlocks = 3;
    tilingData->tileL2Tiling.nTileBlocks = 16;
    tilingData->tileTiling.M = 512;
    tilingData->tileTiling.N = 8192;
    tilingData->tileTiling.Ka = 1024;
    tilingData->tileTiling.Kb = 1024;
    tilingData->tileTiling.singleCoreM = 128;
    tilingData->tileTiling.singleCoreN = 256;
    tilingData->tileTiling.singleCoreK = 1024;
    tilingData->tileTiling.baseM = 128;
    tilingData->tileTiling.baseN = 256;
    tilingData->tileTiling.baseK = 64;
    tilingData->tileTiling.stepKa = 4;
    tilingData->tileTiling.stepKb = 4;
    tilingData->tileTiling.isBias = 0;
    tilingData->tileTiling.depthA1 = 8;
    tilingData->tileTiling.depthB1 = 8;
    tilingData->tileTiling.stepM = 1;
    tilingData->tileTiling.stepN = 1;
    tilingData->tileTiling.usedCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(512 * 1024 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(8192 * 1024 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(512 * 8192 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, biasGM, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}

TEST_F(AllGatherMatmulTest, AllGatherMatmulTestComputationOnly)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, nullptr, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}

TEST_F(AllGatherMatmulTest, AllGatherMatmulTestCommunicationOnly)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 16;
    tilingData->param.aicCoreNum = 20;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, nullptr, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}

TEST_F(AllGatherMatmulTest, AllGatherMatmulTestNoBiasNormalization)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    Mc2Tiling::AllGatherMatmulTilingData *tilingData = reinterpret_cast<Mc2Tiling::AllGatherMatmulTilingData *>(tiling);
    tilingData->param.tailM = 372;
    tilingData->param.aicCoreNum = 20;
    tilingData->param.rankDim = 8;
    tilingData->param.rankM = 1012;
    tilingData->param.tileCnt = 5;
    tilingData->param.rankN = 768;
    tilingData->param.rankK = 6144;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1012 * 6144 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(768 * 6144 * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1012 * 768 * sizeof(uint16_t));
    uint8_t *aicpuWin = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024 * sizeof(uint8_t));

    auto allGatherMatmulWrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                     GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        all_gather_matmul<true, false, false>(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(allGatherMatmulWrapper, 20, aGM, bGM, nullptr, output, nullptr, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)aicpuWin);
}
