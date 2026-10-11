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
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#include "batch_mat_mul_reduce_scatter_allto_all_tiling_def.h"
#include "../../../op_kernel/batch_mat_mul_reduce_scatter_allto_all.cpp"

class BatchMatmulReduceScatterAllToAllTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "batch_mat_mul_reduce_scatter_allto_all_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "batch_mat_mul_reduce_scatter_allto_all_test TearDown\n" << std::endl;
    }
};

TEST_F(BatchMatmulReduceScatterAllToAllTest, BatchMatmulReduceScatterAllToAllFp16NoBias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t E = 8;
    size_t C = 2;
    size_t H = 8;
    size_t M = 8;
    size_t ep = 4;
    size_t tp = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = E * C * H + 2 * E * C / tp * H;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatMulReduceScatterAlltoAllTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    BatchMatMulReduceScatterAlltoAllTilingData* tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = numBlocks * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    auto batchMatMulReduceScatterAlltoAllWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        batch_mat_mul_reduce_scatter_allto_all<1, false, false, false>(xGM, weightGM, biasGM, yGM, workspaceGM,
                                                                       tilingGM);
    };
    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, nullptr, yGM, workspace, tiling);

    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(BatchMatmulReduceScatterAllToAllTest, BatchMatmulReduceScatterAllToAllFp16WithFp16Bias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t E = 8;
    size_t C = 2;
    size_t H = 8;
    size_t M = 8;
    size_t ep = 4;
    size_t tp = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = E * C * H + 2 * E * C / tp * H;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatMulReduceScatterAlltoAllTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    BatchMatMulReduceScatterAlltoAllTilingData* tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = numBlocks * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * sizeof(uint16_t));
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    auto batchMatMulReduceScatterAlltoAllWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        batch_mat_mul_reduce_scatter_allto_all<1, false, false, false>(xGM, weightGM, biasGM, yGM, workspaceGM,
                                                                       tilingGM);
    };
    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, biasGM, yGM, workspace, tiling);

    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, biasGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(BatchMatmulReduceScatterAllToAllTest, BatchMatmulReduceScatterAllToAllBf16NoBias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'bfloat16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t E = 8;
    size_t C = 2;
    size_t H = 8;
    size_t M = 8;
    size_t ep = 4;
    size_t tp = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = E * C * H + 2 * E * C / tp * H;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatMulReduceScatterAlltoAllTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    BatchMatMulReduceScatterAlltoAllTilingData* tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = numBlocks * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    auto batchMatMulReduceScatterAlltoAllWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        batch_mat_mul_reduce_scatter_allto_all<1, false, false, false>(xGM, weightGM, biasGM, yGM, workspaceGM,
                                                                       tilingGM);
    };
    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, nullptr, yGM, workspace, tiling);

    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(BatchMatmulReduceScatterAllToAllTest, BatchMatmulReduceScatterAllToAllBf16WithFloatBias)
{
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'bfloat16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t E = 8;
    size_t C = 2;
    size_t H = 8;
    size_t M = 8;
    size_t ep = 4;
    size_t tp = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = E * C * H + 2 * E * C / tp * H;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatMulReduceScatterAlltoAllTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    BatchMatMulReduceScatterAlltoAllTilingData* tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = numBlocks * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(E / ep * H * sizeof(float));
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    auto batchMatMulReduceScatterAlltoAllWarpper = [](GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        batch_mat_mul_reduce_scatter_allto_all<1, false, false, false>(xGM, weightGM, biasGM, yGM, workspaceGM,
                                                                       tilingGM);
    };
    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, biasGM, yGM, workspace, tiling);

    ICPU_RUN_KF(batchMatMulReduceScatterAlltoAllWarpper, numBlocks, xGM, weightGM, biasGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
}
