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
#include "moe_distribute_dispatch_tiling_def.h"

extern "C" __global__ __aicore__ void moe_distribute_dispatch(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

class moe_distribute_dispatch_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "moe_distribute_dispatch_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "moe_distribute_dispatch_test TearDown\n" << std::endl;
    }
};

TEST_F(moe_distribute_dispatch_test, moe_distribute_dispatch_test_1000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchTilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchTilingData*>(tiling);
    tiling_data->moeDistributeDispatchInfo.epWorldSize = 8;
    tiling_data->moeDistributeDispatchInfo.tpWorldSize = 2;
    tiling_data->moeDistributeDispatchInfo.epRankId = 0;
    tiling_data->moeDistributeDispatchInfo.tpRankId = 0;
    tiling_data->moeDistributeDispatchInfo.expertShardType = 0;
    tiling_data->moeDistributeDispatchInfo.sharedExpertRankNum = 1;
    tiling_data->moeDistributeDispatchInfo.moeExpertNum = 7;
    tiling_data->moeDistributeDispatchInfo.quantMode = 0;
    tiling_data->moeDistributeDispatchInfo.globalBs = 64;
    tiling_data->moeDistributeDispatchInfo.bs = 8;
    tiling_data->moeDistributeDispatchInfo.k = 7;
    tiling_data->moeDistributeDispatchInfo.h = 7168;
    tiling_data->moeDistributeDispatchInfo.aivNum = 48;
    tiling_data->moeDistributeDispatchInfo.isQuant = false;
    tiling_data->moeDistributeDispatchInfo.reserved1 = false;
    tiling_data->moeDistributeDispatchInfo.reserved2 = false;
    tiling_data->moeDistributeDispatchInfo.reserved3 = false;
    tiling_data->moeDistributeDispatchInfo.totalUbSize = 196352;


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(moe_distribute_dispatch, 48, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
}

//MoeDistributeDispatchA2 test do dispatch unquant kernel
TEST_F(moe_distribute_dispatch_test, moe_distribute_dispatch_test_2000001000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchA2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchInfo.epWorldSize = 8;
    tiling_data->moeDistributeDispatchInfo.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.epRankId = 0;
    tiling_data->moeDistributeDispatchInfo.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.expertSharedType = 0;
    tiling_data->moeDistributeDispatchInfo.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchInfo.quantMode = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.globalBs = 64;
    tiling_data->moeDistributeDispatchInfo.bs = 8;
    tiling_data->moeDistributeDispatchInfo.k = 7;
    tiling_data->moeDistributeDispatchInfo.h = 7168;
    tiling_data->moeDistributeDispatchInfo.aivNum = 40;//??
    tiling_data->moeDistributeDispatchInfo.isTokenMask = false;
    tiling_data->moeDistributeDispatchInfo.reserved1 = false;
    tiling_data->moeDistributeDispatchInfo.reserved2 = false;
    tiling_data->moeDistributeDispatchInfo.reserved3 = false;
    tiling_data->moeDistributeDispatchInfo.totalUbSize = 196352;//??


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(2000001000);
    ICPU_RUN_KF(moe_distribute_dispatch, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
}

//MoeDistributeDispatchA2 test do dispatch int8 quant kernel
TEST_F(moe_distribute_dispatch_test, moe_distribute_dispatch_test_2000001002) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchA2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchInfo.epWorldSize = 8;
    tiling_data->moeDistributeDispatchInfo.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.epRankId = 0;
    tiling_data->moeDistributeDispatchInfo.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.expertSharedType = 0;
    tiling_data->moeDistributeDispatchInfo.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchInfo.quantMode = 2;
    tiling_data->moeDistributeDispatchInfo.globalBs = 64;
    tiling_data->moeDistributeDispatchInfo.bs = 8;
    tiling_data->moeDistributeDispatchInfo.k = 7;
    tiling_data->moeDistributeDispatchInfo.h = 7168;
    tiling_data->moeDistributeDispatchInfo.aivNum = 40;
    tiling_data->moeDistributeDispatchInfo.isTokenMask = false;
    tiling_data->moeDistributeDispatchInfo.reserved1 = false;
    tiling_data->moeDistributeDispatchInfo.reserved2 = false;
    tiling_data->moeDistributeDispatchInfo.reserved3 = false;
    tiling_data->moeDistributeDispatchInfo.totalUbSize = 196352;


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(2000001002);
    ICPU_RUN_KF(moe_distribute_dispatch, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
}

//MoeDistributeDispatchA2 test do dispatch int8 quant kernel with smooth scale
TEST_F(moe_distribute_dispatch_test, moe_distribute_dispatch_test_2000001012) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchA2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchInfo.epWorldSize = 8;
    tiling_data->moeDistributeDispatchInfo.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.epRankId = 0;
    tiling_data->moeDistributeDispatchInfo.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.expertSharedType = 0;
    tiling_data->moeDistributeDispatchInfo.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchInfo.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchInfo.quantMode = 1;//
    tiling_data->moeDistributeDispatchInfo.globalBs = 64;
    tiling_data->moeDistributeDispatchInfo.bs = 8;
    tiling_data->moeDistributeDispatchInfo.k = 7;
    tiling_data->moeDistributeDispatchInfo.h = 7168;
    tiling_data->moeDistributeDispatchInfo.aivNum = 40;//??
    tiling_data->moeDistributeDispatchInfo.isTokenMask = false;
    tiling_data->moeDistributeDispatchInfo.reserved1 = false;
    tiling_data->moeDistributeDispatchInfo.reserved2 = false;
    tiling_data->moeDistributeDispatchInfo.reserved3 = false;
    tiling_data->moeDistributeDispatchInfo.totalUbSize = 192 * 1024;//??


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(2000001012);
    ICPU_RUN_KF(moe_distribute_dispatch, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
}