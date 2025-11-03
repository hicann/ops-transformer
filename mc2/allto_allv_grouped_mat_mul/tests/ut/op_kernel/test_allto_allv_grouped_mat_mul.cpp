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
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "allto_allv_grouped_mat_mul_tiling_def.h"

extern "C" __global__ __aicore__ void allto_allv_grouped_mat_mul(GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM);

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class allto_allv_grouped_mat_mul_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "allto_allv_grouped_mat_mul_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "allto_allv_grouped_mat_mul_test TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_10)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_100)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_101)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_110)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(110);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_111)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_1000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_1010)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1010);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_1110)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1110);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(allto_allv_grouped_mat_mul_test, allto_allv_grouped_mat_mul_test_1111)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tiling_data = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1111);
    ICPU_RUN_KF(allto_allv_grouped_mat_mul, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}