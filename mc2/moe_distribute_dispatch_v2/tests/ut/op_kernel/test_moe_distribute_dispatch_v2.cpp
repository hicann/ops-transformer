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
#include "moe_distribute_dispatch_v2_tiling_def.h"

extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoForCombine,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);


struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

class moe_distribute_dispatch_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "moe_distribute_dispatch_v2_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "moe_distribute_dispatch_v2_test TearDown\n" << std::endl;
    }
};

TEST_F(moe_distribute_dispatch_v2_test, moe_distribute_dispatch_v2_test_1000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchV2Info.epWorldSize = 8;
    tiling_data->moeDistributeDispatchV2Info.tpWorldSize = 2;
    tiling_data->moeDistributeDispatchV2Info.epRankId = 0;
    tiling_data->moeDistributeDispatchV2Info.tpRankId = 0;
    tiling_data->moeDistributeDispatchV2Info.expertShardType = 0;
    tiling_data->moeDistributeDispatchV2Info.sharedExpertRankNum = 1;
    tiling_data->moeDistributeDispatchV2Info.moeExpertNum = 7;
    tiling_data->moeDistributeDispatchV2Info.quantMode = 0;
    tiling_data->moeDistributeDispatchV2Info.globalBs = 64;
    tiling_data->moeDistributeDispatchV2Info.bs = 8;
    tiling_data->moeDistributeDispatchV2Info.k = 7;
    tiling_data->moeDistributeDispatchV2Info.h = 7168;
    tiling_data->moeDistributeDispatchV2Info.aivNum = 48;
    tiling_data->moeDistributeDispatchV2Info.isTokenMask= false;
    tiling_data->moeDistributeDispatchV2Info.totalUbSize = 196352;

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
    ICPU_RUN_KF(moe_distribute_dispatch_v2, 48, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

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
TEST_F(moe_distribute_dispatch_v2_test, moe_distribute_dispatch_v2_test_2000001000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchV2Info.epWorldSize = 8;
    tiling_data->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.epRankId = 0;
    tiling_data->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.expertShardType = 0;
    tiling_data->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchV2Info.quantMode = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.globalBs = 64;
    tiling_data->moeDistributeDispatchV2Info.bs = 8;
    tiling_data->moeDistributeDispatchV2Info.k = 7;
    tiling_data->moeDistributeDispatchV2Info.h = 7168;
    tiling_data->moeDistributeDispatchV2Info.aivNum = 40;//??
    tiling_data->moeDistributeDispatchV2Info.isTokenMask= false;
    tiling_data->moeDistributeDispatchV2Info.totalUbSize = 196352;//??

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
    ICPU_RUN_KF(moe_distribute_dispatch_v2, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

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
TEST_F(moe_distribute_dispatch_v2_test, moe_distribute_dispatch_v2_test_2000001002) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchV2Info.epWorldSize = 8;
    tiling_data->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.epRankId = 0;
    tiling_data->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.expertShardType = 0;
    tiling_data->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchV2Info.quantMode = 2;
    tiling_data->moeDistributeDispatchV2Info.globalBs = 64;
    tiling_data->moeDistributeDispatchV2Info.bs = 8;
    tiling_data->moeDistributeDispatchV2Info.k = 7;
    tiling_data->moeDistributeDispatchV2Info.h = 7168;
    tiling_data->moeDistributeDispatchV2Info.aivNum = 40;
    tiling_data->moeDistributeDispatchV2Info.isTokenMask= false;
    tiling_data->moeDistributeDispatchV2Info.totalUbSize = 196352;

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
    ICPU_RUN_KF(moe_distribute_dispatch_v2, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

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
TEST_F(moe_distribute_dispatch_v2_test, moe_distribute_dispatch_v2_test_2000001012) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tiling_data = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tiling_data->moeDistributeDispatchV2Info.epWorldSize = 8;
    tiling_data->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.epRankId = 0;
    tiling_data->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.expertShardType = 0;
    tiling_data->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tiling_data->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tiling_data->moeDistributeDispatchV2Info.quantMode = 1;//
    tiling_data->moeDistributeDispatchV2Info.globalBs = 64;
    tiling_data->moeDistributeDispatchV2Info.bs = 8;
    tiling_data->moeDistributeDispatchV2Info.k = 7;
    tiling_data->moeDistributeDispatchV2Info.h = 7168;
    tiling_data->moeDistributeDispatchV2Info.aivNum = 40;//??
    tiling_data->moeDistributeDispatchV2Info.isTokenMask= false;
    tiling_data->moeDistributeDispatchV2Info.totalUbSize = 192 * 1024;//??

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
    ICPU_RUN_KF(moe_distribute_dispatch_v2, 40, x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspace, tiling);

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