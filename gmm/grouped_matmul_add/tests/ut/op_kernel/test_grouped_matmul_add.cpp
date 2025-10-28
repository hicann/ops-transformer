/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "grouped_matmul_add_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void grouped_matmul_add(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y, GM_ADDR yRef, GM_ADDR workspace, GM_ADDR tiling);
class grouped_matmul_add_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_matmul_add_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_matmul_add_test TearDown\n" << std::endl;
    }
};

TEST_F(grouped_matmul_add_test, test_case_bf16)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t xSize = 1280 * 345 * sizeof(half);
    size_t weightSize = 1280 * 567 * sizeof(half);
    size_t groupedListSize = 1 * 2 * sizeof(int64_t);
    size_t ySize = 345 * 2 * 567 * sizeof(float);
    size_t tilingSize = sizeof(GroupedMatmulAddTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* groupedList = (uint8_t*)AscendC::GmAlloc(groupedListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 40;

    GroupedMatmulAddTilingData* tilingData = reinterpret_cast<GroupedMatmulAddTilingData*>(tiling);
    tilingData->gmmBaseParams.groupNum = 2;
    tilingData->gmmBaseParams.coreNum = blockDim;
    tilingData->gmmBaseParams.groupType = 2;

    tilingData->gmmArray.mList[0] = 345;
    tilingData->gmmArray.kList[0] = -1;
    tilingData->gmmArray.nList[0] = 567;

    tilingData->mmTilingData.usedCoreNum = 1;
    tilingData->mmTilingData.M = 345;
    tilingData->mmTilingData.N = 567;
    tilingData->mmTilingData.Ka = 1280;
    tilingData->mmTilingData.Kb = 1280;
    tilingData->mmTilingData.singleCoreM = 345;
    tilingData->mmTilingData.singleCoreN = 256;
    tilingData->mmTilingData.singleCoreK = 1280;
    tilingData->mmTilingData.baseM = 128;
    tilingData->mmTilingData.baseN = 256;
    tilingData->mmTilingData.baseK = 64;
    tilingData->mmTilingData.depthA1 = 8;
    tilingData->mmTilingData.depthB1 = 8;
    tilingData->mmTilingData.stepM = 1;
    tilingData->mmTilingData.stepN = 1;
    tilingData->mmTilingData.stepKa = 4;
    tilingData->mmTilingData.stepKb = 4;
    tilingData->mmTilingData.isBias = 0;
    tilingData->mmTilingData.transLength = 0;
    tilingData->mmTilingData.iterateOrder = 0;
    tilingData->mmTilingData.dbL0A = true;
    tilingData->mmTilingData.dbL0B = true;
    tilingData->mmTilingData.dbL0C = false;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grouped_matmul_add, blockDim, x, weight, groupedList, y, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(groupedList);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grouped_matmul_add_test, test_case_fp16)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t xSize = 1280 * 345 * sizeof(half);
    size_t weightSize = 1280 * 567 * sizeof(half);
    size_t groupedListSize = 1 * 2 * sizeof(int64_t);
    size_t ySize = 345 * 2 * 567 * sizeof(float);
    size_t tilingSize = sizeof(GroupedMatmulAddTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* groupedList = (uint8_t*)AscendC::GmAlloc(groupedListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 40;

    GroupedMatmulAddTilingData* tilingData = reinterpret_cast<GroupedMatmulAddTilingData*>(tiling);
    tilingData->gmmBaseParams.groupNum = 2;
    tilingData->gmmBaseParams.coreNum = blockDim;
    tilingData->gmmBaseParams.groupType = 2;

    tilingData->gmmArray.mList[0] = 345;
    tilingData->gmmArray.kList[0] = -1;
    tilingData->gmmArray.nList[0] = 567;

    tilingData->mmTilingData.usedCoreNum = 1;
    tilingData->mmTilingData.M = 345;
    tilingData->mmTilingData.N = 567;
    tilingData->mmTilingData.Ka = 1280;
    tilingData->mmTilingData.Kb = 1280;
    tilingData->mmTilingData.singleCoreM = 345;
    tilingData->mmTilingData.singleCoreN = 256;
    tilingData->mmTilingData.singleCoreK = 1280;
    tilingData->mmTilingData.baseM = 128;
    tilingData->mmTilingData.baseN = 256;
    tilingData->mmTilingData.baseK = 64;
    tilingData->mmTilingData.depthA1 = 8;
    tilingData->mmTilingData.depthB1 = 8;
    tilingData->mmTilingData.stepM = 1;
    tilingData->mmTilingData.stepN = 1;
    tilingData->mmTilingData.stepKa = 4;
    tilingData->mmTilingData.stepKb = 4;
    tilingData->mmTilingData.isBias = 0;
    tilingData->mmTilingData.transLength = 0;
    tilingData->mmTilingData.iterateOrder = 0;
    tilingData->mmTilingData.dbL0A = true;
    tilingData->mmTilingData.dbL0B = true;
    tilingData->mmTilingData.dbL0C = false;


    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(grouped_matmul_add, blockDim, x, weight, groupedList, y, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(groupedList);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
