/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
// #include "foreach_minimum_scalar_tensorlist.h"

extern "C" __global__ __aicore__ void allto_all_all_gather_batch_mat_mul(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, 
                                                                         GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                                                                         GM_ADDR workspaceGM, GM_ADDR tilingGM);

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class allto_all_all_gather_batch_mat_mul_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "allto_all_all_gather_batch_mat_mul_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "allto_all_all_gather_batch_mat_mul_test TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000001) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000001);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000011) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000011);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000101) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000101);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000111) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000111);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001001) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001001);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001011) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001011);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001101) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001101);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001111) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001111);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002001) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002001);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002011) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002011);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002101) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002101);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002111) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002111);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003001) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003001);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003011) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003011);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003101) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003101);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003111) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003111);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

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
TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000000) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000000);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000010) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000010);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000100) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000100);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100000110) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000110);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001000) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001000);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001010) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001010);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001100) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001100);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100001110) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100001110);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002000) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002000);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002010) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002010);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002100) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002100);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100002110) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100002110);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003000) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003000);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003010) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003010);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003100) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003100);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}

TEST_F(allto_all_all_gather_batch_mat_mul_test, allto_all_all_gather_batch_mat_mul_test_100003110) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./allto_all_all_gather_batch_mat_mul_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t E = 64;
    size_t C = 64;
    size_t H = 128;
    size_t M = 128;
    size_t ep = 4;
    size_t tp = 4;
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = (E * C * H + 2 * E * C * H * tp + (ep - 1) * C * M * tp * E / ep) * 2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllAllGatherBatchMatMulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllAllGatherBatchMatMulTilingData *tiling_data = reinterpret_cast<AlltoAllAllGatherBatchMatMulTilingData*>(tiling);
    tiling_data->commonTiling.epGroupSize = 4;
    tiling_data->commonTiling.tpGroupSize = 4;
    tiling_data->commonTiling.aivCoreNum = blockDim;
    tiling_data->commonTiling.expert = E;
    tiling_data->commonTiling.COverTp = C;
    tiling_data->commonTiling.MOverTp = M;
    tiling_data->commonTiling.H = H;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * H / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * M / tp * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * sizeof(uint16_t));
    uint8_t *y1GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));
    uint8_t *y2GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * H * sizeof(uint16_t));
    uint8_t *y3GM = (uint8_t *)AscendC::GmAlloc(E / ep * ep * C * M / tp * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100003110);
    ICPU_RUN_KF(allto_all_all_gather_batch_mat_mul, 20, xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)y1GM);
    AscendC::GmFree((void*)y2GM);
    AscendC::GmFree((void*)y3GM);
}