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
 * \file test_matmul_all_reduce_add_rms_norm.cpp
 * \brief
 */

#include <unistd.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"

using namespace std;

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include <iostream>
#include <string>
#include "data_utils.h"
#include "matmul_all_reduce_add_rms_norm_tiling_def.h"
#include "string.h"
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

extern "C" __global__ __aicore__ void matmul_all_reduce_add_rms_norm(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                     GM_ADDR residualGM, GM_ADDR gammaGM,
                                                                     GM_ADDR antiquantScaleGM,
                                                                     GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                                                     GM_ADDR yGM, GM_ADDR normOutGM,
                                                                     GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern uint8_t* g_hcclContextReserved[2];

struct HcclSignalInfo {
    uint64_t resId;  // 在代表event时为eventid，notify时为notifyid
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
};

struct HcclCombinOpSignalParam {
    HcclSignalInfo noIpcNotifys[8 * 2];
    HcclSignalInfo ipcNotifys[8 * 4];
    HcclSignalInfo noIpcEvents[8];
    HcclSignalInfo aicpuNotify;
    HcclSignalInfo aicpuOpNotify[2]; // 集合通信AICPU展开资源
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
};

struct HcclConfig {
    uint8_t determinism;  // 确定性计算开关
};
struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[8];
    uint64_t windowsOut[8];
    char hcomId[128];
    HcclStreamInfo streamInfo[8];
    HcclCombinOpSignalParam signalInfo;
    HcclConfig config;  // 配置参数
};

class matmul_all_reduce_add_rms_norm_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        cout << "matmul_all_reduce_add_rms_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        cout << "matmul_all_reduce_add_rms_norm_test TearDown\n" << endl;
    }
};

TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_no_bias) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(10000000000000001100UL);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, nullptr, residualGM, gammaGM,  nullptr, nullptr, nullptr,
                y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}

TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_bias) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;
    tilingData->matmulAllReduceTilingData.param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(10000000000000000000UL);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
                nullptr, nullptr, nullptr, y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}

TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_11000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;
    tilingData->matmulAllReduceTilingData.param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(11000);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
    nullptr, nullptr, nullptr, y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}

TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_11100) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;
    tilingData->matmulAllReduceTilingData.param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(11100);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
    nullptr, nullptr, nullptr, y, output, nullptr, tiling);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
    nullptr, nullptr, nullptr, y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}
TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_1111) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;
    tilingData->matmulAllReduceTilingData.param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(1111);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
    nullptr, nullptr, nullptr, y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}

TEST_F(matmul_all_reduce_add_rms_norm_test, matmul_all_reduce_add_rms_norm_test_1011) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAllReduceAddRmsNormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MatmulAllReduceAddRmsNormTilingData *tilingData = reinterpret_cast<MatmulAllReduceAddRmsNormTilingData *>(tiling);
    tilingData->matmulAllReduceTilingData.param.tailM = 16;
    tilingData->matmulAllReduceTilingData.param.aicCoreNum = 1;
    tilingData->matmulAllReduceTilingData.param.biasLen = 1536 * sizeof(float);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(1024 * 12288 * sizeof(uint16_t));
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(1536 * 12288 * sizeof(uint16_t));
    uint8_t *residualGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *gammaGM = (uint8_t *)AscendC::GmAlloc(1024 * 1536 * sizeof(uint16_t));
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(1536 * sizeof(uint16_t));
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1536 * 1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(matmul_all_reduce_add_rms_norm, 1, aGM, bGM, biasGM, residualGM, gammaGM,
    nullptr, nullptr, nullptr, y, output, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)aGM);
    AscendC::GmFree((void *)bGM);
    AscendC::GmFree((void *)residualGM);
    AscendC::GmFree((void *)gammaGM);
    AscendC::GmFree((void *)biasGM);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)y);
}
