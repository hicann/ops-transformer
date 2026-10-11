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
#include "allto_all_all_gather_batch_mat_mul_tiling_def.h"
#include "../../../op_kernel/allto_all_all_gather_batch_mat_mul.cpp"

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class AlltoAllAllGatherBatchMatMulTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllAllGatherBatchMatMulTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AlltoAllAllGatherBatchMatMulTest TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest1)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, false, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                          workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest5)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, false, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest9)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, true, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest13)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, true, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest17)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, false, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest21)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, false, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest25)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, true, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(13);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest29)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, true, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(15);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest33)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, false, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(17);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest37)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, false, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(19);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest41)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, true, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest45)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, true, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(23);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest49)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, false, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(25);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest53)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, false, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(27);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest57)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, false, true, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(29);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest61)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<1, true, true, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                      workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(31);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

// shard = 0
TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest0)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, false, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                          workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest4)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, false, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest8)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, true, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest12)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, true, false, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest16)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, false, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest20)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, false, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest24)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, true, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest28)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, true, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(14);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest32)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, false, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                         workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(16);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest36)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, false, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(18);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest40)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, true, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest44)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, true, false, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest48)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, false, false, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                        workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(24);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest52)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, false, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(26);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest56)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, true, true, false>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                       workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(28);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(AlltoAllAllGatherBatchMatMulTest, AlltoAllAllGatherBatchMatMulTest60)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData* tilingData =
        reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tilingData->commonTiling.epGroupSize = 4;
    tilingData->commonTiling.tpGroupSize = 4;
    tilingData->commonTiling.aivCoreNum = numBlocks;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.COverTp = C;
    tilingData->commonTiling.MOverTp = M;
    tilingData->commonTiling.H = H;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t* y1GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t* y2GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t* y3GM = (uint8_t*)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    auto alltoAllAllGatherBatchMatMulWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                  GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        allto_all_all_gather_batch_mat_mul<0, true, true, true, true>(xGM, weightGM, biasGM, y1GM, y2GM, y3GM,
                                                                      workspaceGM, tilingGM);
    };
    ICPU_SET_TILING_KEY(30);
    ICPU_RUN_KF(alltoAllAllGatherBatchMatMulWarpper, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}
