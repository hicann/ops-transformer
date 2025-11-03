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
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "grouped_mat_mul_allto_allv_tiling_def.h"

extern "C" __global__ __aicore__ void grouped_mat_mul_allto_allv(GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR sendCountsTensorOptionalGM,
                                                                 GM_ADDR recvCountsTensorOptionalGM,
                                                                 GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                                                 GM_ADDR yGM, GM_ADDR mmyOptionalGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM);

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class grouped_mat_mul_allto_allv_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_mat_mul_allto_allv_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_mat_mul_allto_allv_test TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(grouped_mat_mul_allto_allv_test, grouped_mat_mul_allto_allv_test_0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 10 * 1024 * 1024;
    size_t usrWorkspaceSize = 10 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(GroupedMatMulAlltoAllvTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    GmmAlltoAllvCommonTilingInfo commonTilingInfo{4096, 7168, 7168, 1,     4096,  2048,  4096, 4096,
                                                  1,    20,   40,   false, false, false, false};
    GroupedMatMulAlltoAllvTilingData* tiling_data = reinterpret_cast<GroupedMatMulAlltoAllvTilingData*>(tiling);
    tiling_data->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.H * sizeof(uint16_t));
    uint8_t* gmmweightGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* sendCountsTensorOptionalGM = nullptr;
    uint8_t* recvCountsTensorOptionalGM = nullptr;
    uint8_t* mmxOptionalGM = nullptr;
    uint8_t* mmweightOptionalGM = nullptr;
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BsK * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyOptionalGM = nullptr;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(grouped_mat_mul_allto_allv, 20, gmmxGM, gmmweightGM, sendCountsTensorOptionalGM,
                recvCountsTensorOptionalGM, mmxOptionalGM, mmweightOptionalGM, yGM, mmyOptionalGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)yGM);
}