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
#include "moe_distribute_dispatch_tiling_def.h"
#include "../../../op_kernel/moe_distribute_dispatch.cpp"
#include "../../../op_kernel/moe_distribute_dispatch_tiling_key.h"

class MoeDistributeDispatchTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchTest TearDown\n" << std::endl;
    }
};

TEST_F(MoeDistributeDispatchTest, MoeDistributeDispatchTest1000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchTilingData *tilingData = reinterpret_cast<MoeDistributeDispatchTilingData *>(tiling);
    tilingData->moeDistributeDispatchInfo.epWorldSize = 8;
    tilingData->moeDistributeDispatchInfo.tpWorldSize = 1;
    tilingData->moeDistributeDispatchInfo.epRankId = 0;
    tilingData->moeDistributeDispatchInfo.tpRankId = 0;
    tilingData->moeDistributeDispatchInfo.expertShardType = 0;
    tilingData->moeDistributeDispatchInfo.sharedExpertRankNum = 1;
    tilingData->moeDistributeDispatchInfo.moeExpertNum = 7;
    tilingData->moeDistributeDispatchInfo.quantMode = 0;
    tilingData->moeDistributeDispatchInfo.globalBs = 64;
    tilingData->moeDistributeDispatchInfo.bs = 8;
    tilingData->moeDistributeDispatchInfo.k = 7;
    tilingData->moeDistributeDispatchInfo.h = 7168;
    tilingData->moeDistributeDispatchInfo.aivNum = 48;
    tilingData->moeDistributeDispatchInfo.isQuant = false;
    tilingData->moeDistributeDispatchInfo.reserved1 = false;
    tilingData->moeDistributeDispatchInfo.reserved2 = false;
    tilingData->moeDistributeDispatchInfo.reserved3 = false;
    tilingData->moeDistributeDispatchInfo.totalUbSize = 196352;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchWarrper = [](GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                           GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                           GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                           GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                           GM_ADDR tilingGM) {
        moe_distribute_dispatch<false, TILINGKEY_NO_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE,
                                TILINGKEY_TPL_A3>(x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                                                  dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                                                  tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeDistributeDispatchWarrper, 48, x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)expandXOut);
    AscendC::GmFree((void *)dynamicScalesOut);
    AscendC::GmFree((void *)expandIdxOut);
    AscendC::GmFree((void *)expertTokenNumsOut);
    AscendC::GmFree((void *)epSendCountsOut);
    AscendC::GmFree((void *)tpSendCountsOut);
    AscendC::GmFree((void *)expandScalesOut);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)expertScales);
}

// MoeDistributeDispatchA2 test do dispatch unquant kernel
TEST_F(MoeDistributeDispatchTest, MoeDistributeDispatchTest2000001000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchA2TilingData *>(tiling);
    tilingData->moeDistributeDispatchInfo.epWorldSize = 8;
    tilingData->moeDistributeDispatchInfo.tpWorldSize = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.epRankId = 0;
    tilingData->moeDistributeDispatchInfo.tpRankId = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.expertSharedType = 0;
    tilingData->moeDistributeDispatchInfo.sharedExpertRankNum = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.moeExpertNum = 8;
    tilingData->moeDistributeDispatchInfo.quantMode = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.globalBs = 64;
    tilingData->moeDistributeDispatchInfo.bs = 8;
    tilingData->moeDistributeDispatchInfo.k = 7;
    tilingData->moeDistributeDispatchInfo.h = 7168;
    tilingData->moeDistributeDispatchInfo.aivNum = 40; //??
    tilingData->moeDistributeDispatchInfo.isTokenMask = false;
    tilingData->moeDistributeDispatchInfo.reserved1 = false;
    tilingData->moeDistributeDispatchInfo.reserved2 = false;
    tilingData->moeDistributeDispatchInfo.reserved3 = false;
    tilingData->moeDistributeDispatchInfo.totalUbSize = 196352; //??

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchWarrper = [](GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                           GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                           GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                           GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                           GM_ADDR tilingGM) {
        moe_distribute_dispatch<false, TILINGKEY_NO_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE,
                                TILINGKEY_TPL_A2>(x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                                                  dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                                                  tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeDistributeDispatchWarrper, 40, x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)expandXOut);
    AscendC::GmFree((void *)dynamicScalesOut);
    AscendC::GmFree((void *)expandIdxOut);
    AscendC::GmFree((void *)expertTokenNumsOut);
    AscendC::GmFree((void *)epSendCountsOut);
    AscendC::GmFree((void *)tpSendCountsOut);
    AscendC::GmFree((void *)expandScalesOut);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)expertScales);
}

// MoeDistributeDispatchA2 test do dispatch int8 quant kernel
TEST_F(MoeDistributeDispatchTest, MoeDistributeDispatchTest2000001002)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchA2TilingData *>(tiling);
    tilingData->moeDistributeDispatchInfo.epWorldSize = 8;
    tilingData->moeDistributeDispatchInfo.tpWorldSize = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.epRankId = 0;
    tilingData->moeDistributeDispatchInfo.tpRankId = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.expertSharedType = 0;
    tilingData->moeDistributeDispatchInfo.sharedExpertRankNum = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.moeExpertNum = 8;
    tilingData->moeDistributeDispatchInfo.quantMode = 2;
    tilingData->moeDistributeDispatchInfo.globalBs = 64;
    tilingData->moeDistributeDispatchInfo.bs = 8;
    tilingData->moeDistributeDispatchInfo.k = 7;
    tilingData->moeDistributeDispatchInfo.h = 7168;
    tilingData->moeDistributeDispatchInfo.aivNum = 40;
    tilingData->moeDistributeDispatchInfo.isTokenMask = false;
    tilingData->moeDistributeDispatchInfo.reserved1 = false;
    tilingData->moeDistributeDispatchInfo.reserved2 = false;
    tilingData->moeDistributeDispatchInfo.reserved3 = false;
    tilingData->moeDistributeDispatchInfo.totalUbSize = 196352;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchWarrper = [](GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                           GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                           GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                           GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                           GM_ADDR tilingGM) {
        moe_distribute_dispatch<false, TILINGKEY_PERTOKEN_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE,
                                TILINGKEY_TPL_A2>(x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                                                  dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                                                  tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeDistributeDispatchWarrper, 40, x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)expandXOut);
    AscendC::GmFree((void *)dynamicScalesOut);
    AscendC::GmFree((void *)expandIdxOut);
    AscendC::GmFree((void *)expertTokenNumsOut);
    AscendC::GmFree((void *)epSendCountsOut);
    AscendC::GmFree((void *)tpSendCountsOut);
    AscendC::GmFree((void *)expandScalesOut);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)expertScales);
}

// MoeDistributeDispatchA2 test do dispatch int8 quant kernel with smooth scale
TEST_F(MoeDistributeDispatchTest, MoeDistributeDispatchTest2000001012)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchA2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchA2TilingData *>(tiling);
    tilingData->moeDistributeDispatchInfo.epWorldSize = 8;
    tilingData->moeDistributeDispatchInfo.tpWorldSize = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.epRankId = 0;
    tilingData->moeDistributeDispatchInfo.tpRankId = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.expertSharedType = 0;
    tilingData->moeDistributeDispatchInfo.sharedExpertRankNum = 0; // 针对A2传递默认值 0
    tilingData->moeDistributeDispatchInfo.moeExpertNum = 8;
    tilingData->moeDistributeDispatchInfo.quantMode = 1; //
    tilingData->moeDistributeDispatchInfo.globalBs = 64;
    tilingData->moeDistributeDispatchInfo.bs = 8;
    tilingData->moeDistributeDispatchInfo.k = 7;
    tilingData->moeDistributeDispatchInfo.h = 7168;
    tilingData->moeDistributeDispatchInfo.aivNum = 40; //??
    tilingData->moeDistributeDispatchInfo.isTokenMask = false;
    tilingData->moeDistributeDispatchInfo.reserved1 = false;
    tilingData->moeDistributeDispatchInfo.reserved2 = false;
    tilingData->moeDistributeDispatchInfo.reserved3 = false;
    tilingData->moeDistributeDispatchInfo.totalUbSize = 192 * 1024; //??

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchWarrper = [](GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                           GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                           GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                           GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                           GM_ADDR tilingGM) {
        moe_distribute_dispatch<false, TILINGKEY_PERTOKEN_QUANT, true, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE,
                                TILINGKEY_TPL_A2>(x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                                                  dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                                                  tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeDistributeDispatchWarrper, 40, x, expertIds, scales, xActiveMask, expertScales, expandXOut,
                dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)expandXOut);
    AscendC::GmFree((void *)dynamicScalesOut);
    AscendC::GmFree((void *)expandIdxOut);
    AscendC::GmFree((void *)expertTokenNumsOut);
    AscendC::GmFree((void *)epSendCountsOut);
    AscendC::GmFree((void *)tpSendCountsOut);
    AscendC::GmFree((void *)expandScalesOut);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)expertScales);
}
