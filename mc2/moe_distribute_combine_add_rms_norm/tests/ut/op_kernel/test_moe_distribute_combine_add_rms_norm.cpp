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
#include "moe_distribute_combine_add_rms_norm_tiling_def.h"
#include "../../../op_kernel/moe_distribute_combine_add_rms_norm.cpp"
class MoeDistributeCombineAddRmsNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormTest TearDown\n" << std::endl;
    }
};

TEST_F(MoeDistributeCombineAddRmsNormTest, MoeDistributeCombineAddRmsNormTest11000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tilingData = reinterpret_cast<MoeDistributeCombineV2TilingData *>(tiling);
    tilingData->moeDistributeCombineV2Info.epWorldSize = 8;
    tilingData->moeDistributeCombineV2Info.tpWorldSize = 2;
    tilingData->moeDistributeCombineV2Info.epRankId = 0;
    tilingData->moeDistributeCombineV2Info.tpRankId = 0;
    tilingData->moeDistributeCombineV2Info.expertShardType = 0;
    tilingData->moeDistributeCombineV2Info.sharedExpertRankNum = 0;
    tilingData->moeDistributeCombineV2Info.moeExpertNum = 7;
    tilingData->moeDistributeCombineV2Info.globalBs = 64;
    tilingData->moeDistributeCombineV2Info.bs = 8;
    tilingData->moeDistributeCombineV2Info.k = 7;
    tilingData->moeDistributeCombineV2Info.h = 7168;
    tilingData->moeDistributeCombineV2Info.aivNum = 48;
    tilingData->moeDistributeCombineV2Info.totalUbSize = 196352;
    tilingData->moeDistributeCombineV2Info.isTokenMask = false;
    tilingData->moeDistributeCombineV2Info.hasSharedExpertX = true;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *residualX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *groupList = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *YOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *rstdOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoForCombine = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *oriX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertV = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(11000);
    auto moeDistributeCombineAddRmsNormWrapper =
        [](GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
           GM_ADDR residualX, GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale,
           GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
           GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
           GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_combine_add_rms_norm<true, false, 0, 1>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, residualX, gamma, tpSendCount,
                xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeCombineAddRmsNormWrapper, 20, expandX, expertIds, assistInfoForCombine, epSendCount,
                scales, residualX, gamma, tpSendCount, xActiveMask, activationScale, weightScale, groupList,
                expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                YOut, rstdOut, XOut, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)expandX);
    AscendC::GmFree((void *)residualX);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)groupList);
    AscendC::GmFree((void *)expandScales);
    AscendC::GmFree((void *)sharedExpertX);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)expandIdx);
    AscendC::GmFree((void *)epSendCount);
    AscendC::GmFree((void *)tpSendCount);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)YOut);
    AscendC::GmFree((void *)rstdOut);
    AscendC::GmFree((void *)XOut);
    AscendC::GmFree((void *)winAddr);
    AscendC::GmFree((void *)assistInfoForCombine);
    AscendC::GmFree((void *)elasticInfo);
    AscendC::GmFree((void *)oriX);
    AscendC::GmFree((void *)constExpertAlpha1);
    AscendC::GmFree((void *)constExpertAlpha2);
    AscendC::GmFree((void *)constExpertV);
}

TEST_F(MoeDistributeCombineAddRmsNormTest, MoeDistributeCombineAddRmsNormTest10100)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tilingData = reinterpret_cast<MoeDistributeCombineV2TilingData *>(tiling);
    tilingData->moeDistributeCombineV2Info.epWorldSize = 16;
    tilingData->moeDistributeCombineV2Info.tpWorldSize = 1;
    tilingData->moeDistributeCombineV2Info.epRankId = 0;
    tilingData->moeDistributeCombineV2Info.tpRankId = 0;
    tilingData->moeDistributeCombineV2Info.expertShardType = 0;
    tilingData->moeDistributeCombineV2Info.sharedExpertRankNum = 0;
    tilingData->moeDistributeCombineV2Info.moeExpertNum = 7;
    tilingData->moeDistributeCombineV2Info.globalBs = 64;
    tilingData->moeDistributeCombineV2Info.bs = 8;
    tilingData->moeDistributeCombineV2Info.k = 7;
    tilingData->moeDistributeCombineV2Info.h = 7168;
    tilingData->moeDistributeCombineV2Info.aivNum = 48;
    tilingData->moeDistributeCombineV2Info.totalUbSize = 196352;
    tilingData->moeDistributeCombineV2Info.isTokenMask = false;
    tilingData->moeDistributeCombineV2Info.hasSharedExpertX = true;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *residualX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *groupList = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *YOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *rstdOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoForCombine = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *oriX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertV = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(10100);
    auto moeDistributeCombineAddRmsNormWrapper =
        [](GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
           GM_ADDR residualX, GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale,
           GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
           GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
           GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_combine_add_rms_norm<false, true, 0, 1>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, residualX, gamma, tpSendCount,
                xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeCombineAddRmsNormWrapper, 20, expandX, expertIds, assistInfoForCombine, epSendCount,
                scales, residualX, gamma, tpSendCount, xActiveMask, activationScale, weightScale, groupList,
                expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                YOut, rstdOut, XOut, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)expandX);
    AscendC::GmFree((void *)residualX);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)groupList);
    AscendC::GmFree((void *)expandScales);
    AscendC::GmFree((void *)sharedExpertX);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)expandIdx);
    AscendC::GmFree((void *)epSendCount);
    AscendC::GmFree((void *)tpSendCount);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)YOut);
    AscendC::GmFree((void *)rstdOut);
    AscendC::GmFree((void *)XOut);
    AscendC::GmFree((void *)winAddr);
    AscendC::GmFree((void *)assistInfoForCombine);
    AscendC::GmFree((void *)elasticInfo);
    AscendC::GmFree((void *)oriX);
    AscendC::GmFree((void *)constExpertAlpha1);
    AscendC::GmFree((void *)constExpertAlpha2);
    AscendC::GmFree((void *)constExpertV);
}

TEST_F(MoeDistributeCombineAddRmsNormTest, MoeDistributeCombineAddRmsNormTest10000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tilingData = reinterpret_cast<MoeDistributeCombineV2TilingData *>(tiling);
    tilingData->moeDistributeCombineV2Info.epWorldSize = 16;
    tilingData->moeDistributeCombineV2Info.tpWorldSize = 1;
    tilingData->moeDistributeCombineV2Info.epRankId = 0;
    tilingData->moeDistributeCombineV2Info.tpRankId = 0;
    tilingData->moeDistributeCombineV2Info.expertShardType = 0;
    tilingData->moeDistributeCombineV2Info.sharedExpertRankNum = 0;
    tilingData->moeDistributeCombineV2Info.moeExpertNum = 7;
    tilingData->moeDistributeCombineV2Info.globalBs = 64;
    tilingData->moeDistributeCombineV2Info.bs = 8;
    tilingData->moeDistributeCombineV2Info.k = 7;
    tilingData->moeDistributeCombineV2Info.h = 7168;
    tilingData->moeDistributeCombineV2Info.aivNum = 48;
    tilingData->moeDistributeCombineV2Info.totalUbSize = 196352;
    tilingData->moeDistributeCombineV2Info.isTokenMask = false;
    tilingData->moeDistributeCombineV2Info.hasSharedExpertX = true;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *residualX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *groupList = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *YOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *rstdOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoForCombine = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *oriX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertV = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(10000);
    auto moeDistributeCombineAddRmsNormWrapper =
        [](GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
           GM_ADDR residualX, GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale,
           GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
           GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
           GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_combine_add_rms_norm<false, false, 0, 1>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, residualX, gamma, tpSendCount,
                xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeCombineAddRmsNormWrapper, 20, expandX, expertIds, assistInfoForCombine, epSendCount,
                scales, residualX, gamma, tpSendCount, xActiveMask, activationScale, weightScale, groupList,
                expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                YOut, rstdOut, XOut, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)expandX);
    AscendC::GmFree((void *)residualX);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)groupList);
    AscendC::GmFree((void *)expandScales);
    AscendC::GmFree((void *)sharedExpertX);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)expandIdx);
    AscendC::GmFree((void *)epSendCount);
    AscendC::GmFree((void *)tpSendCount);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)YOut);
    AscendC::GmFree((void *)rstdOut);
    AscendC::GmFree((void *)XOut);
    AscendC::GmFree((void *)winAddr);
    AscendC::GmFree((void *)assistInfoForCombine);
    AscendC::GmFree((void *)elasticInfo);
    AscendC::GmFree((void *)oriX);
    AscendC::GmFree((void *)constExpertAlpha1);
    AscendC::GmFree((void *)constExpertAlpha2);
    AscendC::GmFree((void *)constExpertV);
}

TEST_F(MoeDistributeCombineAddRmsNormTest, MoeDistributeCombineAddRmsNormTest11100)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeCombineV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeDistributeCombineV2TilingData *tilingData = reinterpret_cast<MoeDistributeCombineV2TilingData *>(tiling);
    tilingData->moeDistributeCombineV2Info.epWorldSize = 8;
    tilingData->moeDistributeCombineV2Info.tpWorldSize = 2;
    tilingData->moeDistributeCombineV2Info.epRankId = 0;
    tilingData->moeDistributeCombineV2Info.tpRankId = 0;
    tilingData->moeDistributeCombineV2Info.expertShardType = 0;
    tilingData->moeDistributeCombineV2Info.sharedExpertRankNum = 0;
    tilingData->moeDistributeCombineV2Info.moeExpertNum = 7;
    tilingData->moeDistributeCombineV2Info.globalBs = 64;
    tilingData->moeDistributeCombineV2Info.bs = 8;
    tilingData->moeDistributeCombineV2Info.k = 7;
    tilingData->moeDistributeCombineV2Info.h = 7168;
    tilingData->moeDistributeCombineV2Info.aivNum = 48;
    tilingData->moeDistributeCombineV2Info.totalUbSize = 196352;
    tilingData->moeDistributeCombineV2Info.isTokenMask = false;
    tilingData->moeDistributeCombineV2Info.hasSharedExpertX = true;

    uint8_t *expandX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *residualX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *groupList = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *sharedExpertX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCount = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *YOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *rstdOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *XOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *winAddr = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoForCombine = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *oriX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertAlpha2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *constExpertV = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(11100);
    auto moeDistributeCombineAddRmsNormWrapper =
        [](GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
           GM_ADDR residualX, GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale,
           GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
           GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
           GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_combine_add_rms_norm<true, true, 0, 1>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, residualX, gamma, tpSendCount,
                xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeCombineAddRmsNormWrapper, 20, expandX, expertIds, assistInfoForCombine, epSendCount,
                scales, residualX, gamma, tpSendCount, xActiveMask, activationScale, weightScale, groupList,
                expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                YOut, rstdOut, XOut, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)expandX);
    AscendC::GmFree((void *)residualX);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)xActiveMask);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)groupList);
    AscendC::GmFree((void *)expandScales);
    AscendC::GmFree((void *)sharedExpertX);
    AscendC::GmFree((void *)expertIds);
    AscendC::GmFree((void *)expandIdx);
    AscendC::GmFree((void *)epSendCount);
    AscendC::GmFree((void *)tpSendCount);
    AscendC::GmFree((void *)scales);
    AscendC::GmFree((void *)YOut);
    AscendC::GmFree((void *)rstdOut);
    AscendC::GmFree((void *)XOut);
    AscendC::GmFree((void *)winAddr);
    AscendC::GmFree((void *)assistInfoForCombine);
    AscendC::GmFree((void *)elasticInfo);
    AscendC::GmFree((void *)oriX);
    AscendC::GmFree((void *)constExpertAlpha1);
    AscendC::GmFree((void *)constExpertAlpha2);
    AscendC::GmFree((void *)constExpertV);
}
