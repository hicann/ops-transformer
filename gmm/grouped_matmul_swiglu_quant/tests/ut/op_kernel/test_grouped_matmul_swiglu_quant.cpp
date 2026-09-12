/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "grouped_matmul_swiglu_quant_tiling_def.h"
#include "data_utils.h"

using namespace std;
extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                  GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                  GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, GM_ADDR tiling);
class grouped_matmul_swiglu_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_matmul_swiglu_quant_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_matmul_swiglu_quant_test TearDown\n" << std::endl;
    }
};

TEST_F(grouped_matmul_swiglu_quant_test, test_case_A8W8_tilingkey_0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    int group_num = 2;
    int e = 4;
    int m = 16;
    int k = 256;
    int n = 512;
    size_t xSize = m * k * sizeof(int8_t);
    size_t weightSize = e * k * n * sizeof(int8_t);
    size_t weightScaleSize = e * n * sizeof(float);
    size_t xScaleSize = m * sizeof(float);
    size_t groupedListSize = e * sizeof(int64_t);
    size_t ySize = m * static_cast<int>(n / 2) * sizeof(int8_t);
    size_t yScaleSize = m * sizeof(float);
    size_t tilingSize = sizeof(GMMSwigluQuantTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleSize);
    uint8_t *xScale = (uint8_t *)AscendC::GmAlloc(xScaleSize);
    uint8_t *groupedList = (uint8_t *)AscendC::GmAlloc(groupedListSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *yScale = (uint8_t *)AscendC::GmAlloc(yScaleSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16809984);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint32_t NumBlocks = 20;

    GMMSwigluQuantTilingData *tilingData = reinterpret_cast<GMMSwigluQuantTilingData *>(tiling);
    tilingData->gmmSwigluBaseParams.groupNum = 4;
    tilingData->gmmSwigluBaseParams.coreNum = NumBlocks;
    tilingData->gmmSwigluBaseParams.M = 16;
    tilingData->gmmSwigluBaseParams.K = 256;
    tilingData->gmmSwigluBaseParams.N = 512;
    tilingData->gmmSwigluBaseParams.quantGroupNum = 1;
    tilingData->gmmSwiglu.maxProcessRowNum = 41;
    tilingData->gmmSwiglu.groupListLen = 4;
    tilingData->gmmSwiglu.tokenLen = 512;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(grouped_matmul_swiglu_quant, NumBlocks, x, weight, weightScale, xScale, nullptr, groupedList, y, yScale,
                workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(weightScale);
    AscendC::GmFree(xScale);
    AscendC::GmFree(groupedList);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grouped_matmul_swiglu_quant_test, test_case_A8W8_tilingkey_1)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    int group_num = 2;
    int e = 4;
    int m = 16384;
    int k = 256;
    int n = 4096;
    size_t xSize = m * k * sizeof(int8_t);
    size_t weightSize = e * k * n * sizeof(int8_t);
    size_t weightScaleSize = e * n * sizeof(float);
    size_t xScaleSize = m * sizeof(float);
    size_t groupedListSize = e * sizeof(int64_t);
    size_t ySize = m * static_cast<int>(n / 2) * sizeof(int8_t);
    size_t yScaleSize = m * sizeof(float);
    size_t tilingSize = sizeof(GMMSwigluQuantTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleSize);
    uint8_t *xScale = (uint8_t *)AscendC::GmAlloc(xScaleSize);
    uint8_t *groupedList = (uint8_t *)AscendC::GmAlloc(groupedListSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *yScale = (uint8_t *)AscendC::GmAlloc(yScaleSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16809984);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint32_t NumBlocks = 20;

    GMMSwigluQuantTilingData *tilingData = reinterpret_cast<GMMSwigluQuantTilingData *>(tiling);
    tilingData->gmmSwigluBaseParams.groupNum = 4;
    tilingData->gmmSwigluBaseParams.coreNum = NumBlocks;
    tilingData->gmmSwigluBaseParams.M = 16384;
    tilingData->gmmSwigluBaseParams.K = 256;
    tilingData->gmmSwigluBaseParams.N = 4096;
    tilingData->gmmSwigluBaseParams.quantGroupNum = 1;
    tilingData->gmmSwiglu.maxProcessRowNum = 3;
    tilingData->gmmSwiglu.groupListLen = 4;
    tilingData->gmmSwiglu.tokenLen = 4096;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grouped_matmul_swiglu_quant, NumBlocks, x, weight, weightScale, xScale, nullptr, groupedList, y, yScale,
                workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(weightScale);
    AscendC::GmFree(xScale);
    AscendC::GmFree(groupedList);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
