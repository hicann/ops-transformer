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
#include "distribute_barrier_tiling_def.h"
#include "../../../../common/op_kernel/moe_distribute_base.h"

extern "C" __global__ __aicore__ void distribute_barrier(GM_ADDR xRef, GM_ADDR timeOut, GM_ADDR elasticInfo,
                                                         GM_ADDR xRefOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

extern uint8_t* g_hcclContextReserved[2];

class DistributeBarrierTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclOpResParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        std::cout << "DistributeBarrierInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        std::cout << "DistributeBarrierInfershape TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierTest, DistributeBarrierTest10000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(DistributeBarrierTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto winContext = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
    winContext->localWindowsExp = (uint64_t)workspace;

    DistributeBarrierTilingData* tilingData = reinterpret_cast<DistributeBarrierTilingData*>(tiling);
    tilingData->distributeBarrierInfo.worldSize = 16;
    tilingData->distributeBarrierInfo.rankId = 0;
    tilingData->distributeBarrierInfo.aivNum = 48;

    uint8_t* xRef = (uint8_t*)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t* xRefOut = (uint8_t*)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    // ICPU_RUN_KF(distribute_barrier, 48, xRef, nullptr, nullptr, xRefOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xRef);
    AscendC::GmFree((void*)xRefOut);
}
