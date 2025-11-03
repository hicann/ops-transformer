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
#include "../../../../common/inc/hccl_stub.h"
#include "batch_mat_mul_reduce_scatter_allto_all_tiling_def.h"

extern "C" __global__ __aicore__ void batch_mat_mul_reduce_scatter_allto_all(GM_ADDR xGM, GM_ADDR weightGM,
                                                                             GM_ADDR biasGM, GM_ADDR yGM, 
                                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM);


class batch_matmul_reduce_scatter_all_to_all_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_mat_mul_reduce_scatter_allto_all_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "batch_mat_mul_reduce_scatter_allto_all_test TearDown\n" << std::endl;
    }
};

TEST_F(batch_matmul_reduce_scatter_all_to_all_test, batch_matmul_reduce_scatter_all_to_all_fp16_no_bias) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
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

    BatchMatMulReduceScatterAlltoAllTilingData *tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = blockDim * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000010);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, nullptr, yGM, workspace, tiling);

    ICPU_SET_TILING_KEY(100000110);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(batch_matmul_reduce_scatter_all_to_all_test, batch_matmul_reduce_scatter_all_to_all_fp16_with_fp16_bias) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
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

    BatchMatMulReduceScatterAlltoAllTilingData *tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = blockDim * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * sizeof(uint16_t));
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000011);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, biasGM, yGM, workspace, tiling);

    ICPU_SET_TILING_KEY(100000111);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, biasGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(batch_matmul_reduce_scatter_all_to_all_test, batch_matmul_reduce_scatter_all_to_all_bf16_no_bias) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'bfloat16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
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

    BatchMatMulReduceScatterAlltoAllTilingData *tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = blockDim * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t *biasGM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000010);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, nullptr, yGM, workspace, tiling);

    ICPU_SET_TILING_KEY(100000110);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(batch_matmul_reduce_scatter_all_to_all_test, batch_matmul_reduce_scatter_all_to_all_bf16_with_float_bias) {
    // std::vector<std::vector<uint64_t>> shapeInfos = {{1024, 12288}, {12288, 1536}};
    // system("cd ./batch_matmul_reduce_scatter_all_to_all_data/ && python3 gen_data.py 1024 12288 1536 'bfloat16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
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

    BatchMatMulReduceScatterAlltoAllTilingData *tiling_data =
        reinterpret_cast<BatchMatMulReduceScatterAlltoAllTilingData*>(tiling);
    tiling_data->commonTiling.aivCoreNum = blockDim * 2;
    tiling_data->commonTiling.EOverEp = E / ep;
    tiling_data->commonTiling.C = C;
    tiling_data->commonTiling.H = H;
    tiling_data->commonTiling.MOverTp = M / tp;
    tiling_data->commonTiling.epGroupSize = ep;
    tiling_data->commonTiling.tpGroupSize = tp;
    tiling_data->commonTiling.localTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileCnt = 0;
    tiling_data->commonTiling.domesticTileE.tileLen = 0;

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(E * C * M / tp * sizeof(uint16_t));
    uint8_t *weightGM = (uint8_t *)AscendC::GmAlloc(E / ep * M / tp * H * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(E / ep * H * sizeof(float));
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(E * C / tp * H * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000013);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, biasGM, yGM, workspace, tiling);

    ICPU_SET_TILING_KEY(100000113);
    ICPU_RUN_KF(batch_mat_mul_reduce_scatter_allto_all, blockDim, xGM, weightGM, biasGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)weightGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
}