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
#include "moe_update_expert_tiling_def.h"
#include "../../../op_kernel/moe_update_expert.cpp"
#define GM_ADDR uint8_t*
class MoeUpdateExpertTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeUpdateExpertTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeUpdateExpertTest TearDown\n" << std::endl;
    }
};

// loadbalance by rank
TEST_F(MoeUpdateExpertTest, MoeUpdateExpertTestLbByRank)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeUpdateExpertTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeUpdateExpertTilingData tilingData{128, 8, 256, 8, 48, 0, 8, 0, 0};
    memcpy(tiling, &tilingData, sizeof(MoeUpdateExpertTilingData));
    MoeUpdateExpertTilingData* tilingDataCopy = reinterpret_cast<MoeUpdateExpertTilingData*>(tiling);

    uint8_t* expertIdsGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* eplbTableGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->moeExpertNum * tilingDataCopy->f * sizeof(int32_t));
    uint8_t* expertScalesGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(float));
    uint8_t* pruningThresholdGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->k * sizeof(float));
    uint8_t* activeMaskGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * sizeof(bool));
    uint8_t* balancedExpertIdsOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* balancedActiveMaskOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(bool));

    ICPU_SET_TILING_KEY(0);
    auto moeUpdateExpertWrapper = [](GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM,
                                     GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                     GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        moe_update_expert<TILINGKEY_FLOAT, RANK_ID_BALANCING_MODE>(
            expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM, balancedExpertIdsOutGM,
            balancedActiveMaskOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeUpdateExpertWrapper, 48, expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM,
                balancedExpertIdsOutGM, balancedActiveMaskOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expertIdsGM);
    AscendC::GmFree((void*)eplbTableGM);
    AscendC::GmFree((void*)expertScalesGM);
    AscendC::GmFree((void*)pruningThresholdGM);
    AscendC::GmFree((void*)activeMaskGM);
    AscendC::GmFree((void*)balancedExpertIdsOutGM);
    AscendC::GmFree((void*)balancedActiveMaskOutGM);
}

// loadbalance by token + expert_scales float
TEST_F(MoeUpdateExpertTest, MoeUpdateExpertTestLbByTokenFloat)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeUpdateExpertTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeUpdateExpertTilingData tilingData{128, 8, 256, 8, 48, 0, 8, 0, 1};
    memcpy(tiling, &tilingData, sizeof(MoeUpdateExpertTilingData));
    MoeUpdateExpertTilingData* tilingDataCopy = reinterpret_cast<MoeUpdateExpertTilingData*>(tiling);

    uint8_t* expertIdsGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* eplbTableGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->moeExpertNum * tilingDataCopy->f * sizeof(int32_t));
    uint8_t* expertScalesGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(float));
    uint8_t* pruningThresholdGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->k * sizeof(float));
    uint8_t* activeMaskGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * sizeof(bool));
    uint8_t* balancedExpertIdsOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* balancedActiveMaskOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(bool));

    ICPU_SET_TILING_KEY(1);
    auto moeUpdateExpertWrapper = [](GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM,
                                     GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                     GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        moe_update_expert<TILINGKEY_FLOAT, TOKEN_ID_BALANCING_MODE>(
            expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM, balancedExpertIdsOutGM,
            balancedActiveMaskOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeUpdateExpertWrapper, 48, expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM,
                balancedExpertIdsOutGM, balancedActiveMaskOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expertIdsGM);
    AscendC::GmFree((void*)eplbTableGM);
    AscendC::GmFree((void*)expertScalesGM);
    AscendC::GmFree((void*)pruningThresholdGM);
    AscendC::GmFree((void*)activeMaskGM);
    AscendC::GmFree((void*)balancedExpertIdsOutGM);
    AscendC::GmFree((void*)balancedActiveMaskOutGM);
}

// loadbalance by token + expert_scales half
TEST_F(MoeUpdateExpertTest, MoeUpdateExpertTestLbByTokenHalf)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeUpdateExpertTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeUpdateExpertTilingData tilingData{128, 8, 256, 8, 48, 0, 8, 0, 1};
    memcpy(tiling, &tilingData, sizeof(MoeUpdateExpertTilingData));
    MoeUpdateExpertTilingData* tilingDataCopy = reinterpret_cast<MoeUpdateExpertTilingData*>(tiling);

    uint8_t* expertIdsGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* eplbTableGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->moeExpertNum * tilingDataCopy->f * sizeof(int32_t));
    uint8_t* expertScalesGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(half));
    uint8_t* pruningThresholdGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->k * sizeof(float));
    uint8_t* activeMaskGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * sizeof(bool));
    uint8_t* balancedExpertIdsOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* balancedActiveMaskOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(bool));

    ICPU_SET_TILING_KEY(11);
    auto moeUpdateExpertWrapper = [](GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM,
                                     GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                     GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        moe_update_expert<TILINGKEY_HALF, TOKEN_ID_BALANCING_MODE>(
            expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM, balancedExpertIdsOutGM,
            balancedActiveMaskOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeUpdateExpertWrapper, 48, expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM,
                balancedExpertIdsOutGM, balancedActiveMaskOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expertIdsGM);
    AscendC::GmFree((void*)eplbTableGM);
    AscendC::GmFree((void*)expertScalesGM);
    AscendC::GmFree((void*)pruningThresholdGM);
    AscendC::GmFree((void*)activeMaskGM);
    AscendC::GmFree((void*)balancedExpertIdsOutGM);
    AscendC::GmFree((void*)balancedActiveMaskOutGM);
}

// loadbalance by token + expert_scales bfloat16_t
TEST_F(MoeUpdateExpertTest, MoeUpdateExpertTestLbByTokenBfloat16T)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeUpdateExpertTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeUpdateExpertTilingData tilingData{128, 8, 256, 8, 48, 0, 8, 0, 1};
    memcpy(tiling, &tilingData, sizeof(MoeUpdateExpertTilingData));
    MoeUpdateExpertTilingData* tilingDataCopy = reinterpret_cast<MoeUpdateExpertTilingData*>(tiling);

    uint8_t* expertIdsGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* eplbTableGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->moeExpertNum * tilingDataCopy->f * sizeof(int32_t));
    uint8_t* expertScalesGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(bfloat16_t));
    uint8_t* pruningThresholdGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->k * sizeof(float));
    uint8_t* activeMaskGM = (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * sizeof(bool));
    uint8_t* balancedExpertIdsOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(int32_t));
    uint8_t* balancedActiveMaskOutGM =
        (uint8_t*)AscendC::GmAlloc(tilingDataCopy->bs * tilingDataCopy->k * sizeof(bool));

    ICPU_SET_TILING_KEY(21);
    auto moeUpdateExpertWrapper = [](GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM,
                                     GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                     GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        moe_update_expert<TILINGKEY_BFLOAT16, TOKEN_ID_BALANCING_MODE>(
            expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM, balancedExpertIdsOutGM,
            balancedActiveMaskOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(moeUpdateExpertWrapper, 48, expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM,
                balancedExpertIdsOutGM, balancedActiveMaskOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expertIdsGM);
    AscendC::GmFree((void*)eplbTableGM);
    AscendC::GmFree((void*)expertScalesGM);
    AscendC::GmFree((void*)pruningThresholdGM);
    AscendC::GmFree((void*)activeMaskGM);
    AscendC::GmFree((void*)balancedExpertIdsOutGM);
    AscendC::GmFree((void*)balancedActiveMaskOutGM);
}
