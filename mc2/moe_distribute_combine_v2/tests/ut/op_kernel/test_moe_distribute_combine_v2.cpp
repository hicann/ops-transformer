/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_distribute_combine_v2_tiling_def.h"

extern "C" __global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
      GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

class moe_distribute_combine_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "moe_distribute_combine_v2_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "moe_distribute_combine_v2_test TearDown\n" << std::endl;
    }
};

TEST_F(moe_distribute_combine_v2_test, moe_distribute_combine_v2_test_1000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tiling_data = reinterpret_cast<MoeDistributeCombineV2TilingData*>(tiling);
    tiling_data->moeDistributeCombineV2Info.epWorldSize = 8;
    tiling_data->moeDistributeCombineV2Info.tpWorldSize = 2;
    tiling_data->moeDistributeCombineV2Info.epRankId = 0;
    tiling_data->moeDistributeCombineV2Info.tpRankId = 0;
    tiling_data->moeDistributeCombineV2Info.expertShardType = 0;
    tiling_data->moeDistributeCombineV2Info.sharedExpertRankNum = 1;
    tiling_data->moeDistributeCombineV2Info.moeExpertNum = 7;
    tiling_data->moeDistributeCombineV2Info.globalBs = 64;
    tiling_data->moeDistributeCombineV2Info.bs = 8;
    tiling_data->moeDistributeCombineV2Info.k = 7;
    tiling_data->moeDistributeCombineV2Info.h = 7168;
    tiling_data->moeDistributeCombineV2Info.aivNum = 48;
    tiling_data->moeDistributeCombineV2Info.totalUbSize = 196352;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(moe_distribute_combine_v2, 20, expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, sharedExpertX, XOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expandX);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)expandIdx);
    AscendC::GmFree((void*)epSendCount);
    AscendC::GmFree((void*)tpSendCount);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)XOut);
}

//A2 TEST
TEST_F(moe_distribute_combine_v2_test, moe_distribute_combine_v2_test_2000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tiling_data = reinterpret_cast<MoeDistributeCombineV2TilingData*>(tiling);
    tiling_data->moeDistributeCombineV2Info.epWorldSize = 8;
    tiling_data->moeDistributeCombineV2Info.tpWorldSize = 0;
    tiling_data->moeDistributeCombineV2Info.epRankId = 0;
    tiling_data->moeDistributeCombineV2Info.tpRankId = 0;
    tiling_data->moeDistributeCombineV2Info.expertShardType = 0;
    tiling_data->moeDistributeCombineV2Info.sharedExpertRankNum = 0;
    tiling_data->moeDistributeCombineV2Info.moeExpertNum = 8;
    tiling_data->moeDistributeCombineV2Info.globalBs = 64;
    tiling_data->moeDistributeCombineV2Info.bs = 8;
    tiling_data->moeDistributeCombineV2Info.k = 8;
    tiling_data->moeDistributeCombineV2Info.h = 7168;
    tiling_data->moeDistributeCombineV2Info.aivNum = 40;
    tiling_data->moeDistributeCombineV2Info.totalUbSize = 196352;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(moe_distribute_combine_v2, 20, expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, sharedExpertX, XOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expandX);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)expandIdx);
    AscendC::GmFree((void*)epSendCount);
    AscendC::GmFree((void*)tpSendCount);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)XOut);
}