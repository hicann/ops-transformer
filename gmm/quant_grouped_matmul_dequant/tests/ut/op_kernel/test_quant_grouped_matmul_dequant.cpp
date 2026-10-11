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
#include "gtest/gtest.h"

using namespace std;

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

extern "C" __global__ __aicore__ void quant_grouped_matmul_dequant(GM_ADDR x, GM_ADDR quantized_weight,
                                                                   GM_ADDR weight_scale, GM_ADDR group_list,
                                                                   GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset,
                                                                   GM_ADDR smooth_scale, GM_ADDR y,
                                                                   GM_ADDR usrworkspace, GM_ADDR qmmTiling);

class quant_grouped_matmul_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "quant_grouped_matmul_dequant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "quant_grouped_matmul_dequant_test TearDown\n" << endl;
    }
};

TEST_F(quant_grouped_matmul_dequant_test, test_case_1)
{
    int G = 4;
    int M = 64;
    int K = 256;
    int N = 512;

    int64_t xSize = M * K * sizeof(half);
    int64_t weightSize = G * N * K * sizeof(int8_t);
    int64_t weightScaleSize = G * N * sizeof(float);
    int64_t groupListSize = G * sizeof(int64_t);
    int64_t outSize = M * N * sizeof(half);

#define ORIG_DTYPE_WEIGHT_SCALE DT_FLOAT
#define ORIG_DTYPE_X_SCALE DT_FLOAT

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* weightScaleGM = (uint8_t*)AscendC::GmAlloc(weightScaleSize);
    uint8_t* groupListGM = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* outGM = (uint8_t*)AscendC::GmAlloc(outSize);

    uint64_t* groupListData = reinterpret_cast<uint64_t*>(groupListGM);
    groupListData[0] = 7;
    groupListData[1] = 29;
    groupListData[2] = 31;
    groupListData[3] = 64;

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(QuantMatmulDequantTilingData));
    QuantMatmulDequantTilingData* tilingDatafromBin = reinterpret_cast<QuantMatmulDequantTilingData*>(tiling);
    tilingDatafromBin->CoreNum = 8;
    tilingDatafromBin->perToken = 1;
    tilingDatafromBin->dynamicQuant = 1;
    tilingDatafromBin->smoothScale = 0;
    tilingDatafromBin->originE = 4;
    tilingDatafromBin->originM = 64;
    tilingDatafromBin->originN = 512;
    tilingDatafromBin->originK = 256;
    tilingDatafromBin->originKAligned32 = 256;
    tilingDatafromBin->originKAligned512 = 512;
    tilingDatafromBin->fracN = 32;
    tilingDatafromBin->fracK = 8;

    tilingDatafromBin->dynamicBaseK = 32768;
    tilingDatafromBin->dynamicIterK = 1;
    tilingDatafromBin->dynamicBaseKTail = 256;

    tilingDatafromBin->singleCoreFracN = 4;
    tilingDatafromBin->singleCoreFracNTail = 8;
    tilingDatafromBin->baseFracN = 4;
    tilingDatafromBin->baseFracK = 16;

    tilingDatafromBin->processXKBaseNMax = 1792;
    int64_t workspaceSize = 2115840;
    uint8_t* workspaceGM = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    ICPU_SET_TILING_KEY(10000003);
    ICPU_RUN_KF(quant_grouped_matmul_dequant, 8, xGM, weightGM, weightScaleGM, groupListGM, nullptr, nullptr, nullptr,
                nullptr, outGM, workspaceGM, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(xGM);
    AscendC::GmFree(weightGM);
    AscendC::GmFree(weightScaleGM);
    AscendC::GmFree(groupListGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}

TEST_F(quant_grouped_matmul_dequant_test, test_case_2)
{
    int G = 4;
    int M = 64;
    int K = 256;
    int N = 512;

    int64_t xSize = M * K * sizeof(half);
    int64_t weightSize = G * N * K * sizeof(int8_t);
    int64_t weightScaleSize = G * N * sizeof(float);
    int64_t groupListSize = G * sizeof(int64_t);
    int64_t xScaleSize = 1 * sizeof(float);
    int64_t smoothScaleSize = K * sizeof(half);
    int64_t outSize = M * N * sizeof(half);

#define ORIG_DTYPE_WEIGHT_SCALE DT_FLOAT
#define ORIG_DTYPE_X_SCALE DT_FLOAT

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weightGM = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* weightScaleGM = (uint8_t*)AscendC::GmAlloc(weightScaleSize);
    uint8_t* groupListGM = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* xScaleGM = (uint8_t*)AscendC::GmAlloc(xScaleSize);
    uint8_t* smoothScaleGM = (uint8_t*)AscendC::GmAlloc(smoothScaleSize);
    uint8_t* outGM = (uint8_t*)AscendC::GmAlloc(outSize);

    uint64_t* groupListData = reinterpret_cast<uint64_t*>(groupListGM);
    groupListData[0] = 7;
    groupListData[1] = 29;
    groupListData[2] = 31;
    groupListData[3] = 64;

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(QuantMatmulDequantTilingData));
    QuantMatmulDequantTilingData* tilingDatafromBin = reinterpret_cast<QuantMatmulDequantTilingData*>(tiling);
    tilingDatafromBin->CoreNum = 8;
    tilingDatafromBin->perToken = 0;
    tilingDatafromBin->dynamicQuant = 0;
    tilingDatafromBin->smoothScale = 1;
    tilingDatafromBin->originE = 4;
    tilingDatafromBin->originM = 64;
    tilingDatafromBin->originN = 512;
    tilingDatafromBin->originK = 256;
    tilingDatafromBin->originKAligned32 = 256;
    tilingDatafromBin->originKAligned512 = 512;
    tilingDatafromBin->fracN = 32;
    tilingDatafromBin->fracK = 8;

    tilingDatafromBin->singleCoreFracN = 4;
    tilingDatafromBin->singleCoreFracNTail = 8;
    tilingDatafromBin->baseFracN = 4;
    tilingDatafromBin->baseFracK = 16;

    tilingDatafromBin->processXKBaseNMax = 1792;

    int64_t workspaceSize = 2113792;
    uint8_t* workspaceGM = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    ICPU_SET_TILING_KEY(10000003);
    ICPU_RUN_KF(quant_grouped_matmul_dequant, 8, xGM, weightGM, weightScaleGM, groupListGM, nullptr, xScaleGM, nullptr,
                smoothScaleGM, outGM, workspaceGM, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(xGM);
    AscendC::GmFree(weightGM);
    AscendC::GmFree(weightScaleGM);
    AscendC::GmFree(groupListGM);
    AscendC::GmFree(xScaleGM);
    AscendC::GmFree(smoothScaleGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}
