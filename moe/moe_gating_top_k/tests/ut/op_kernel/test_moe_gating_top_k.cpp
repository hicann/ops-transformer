/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "moe_gating_top_k_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void moe_gating_top_k(
    uint8_t* x, uint8_t* bias, uint8_t* y, uint8_t* expertIdx, uint8_t* out, uint8_t* workspace, uint8_t* tiling);

class moe_gating_top_k_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_gating_top_k_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_gating_top_k_test TearDown\n" << endl;
    }
};

TEST_F(moe_gating_top_k_test, test_case_0)
{
    size_t n = 16;
    size_t k = 8;
    size_t h = 256;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 16;

    size_t x_FileSize = n * h * sizeof(float);
    size_t bias_FileSize = n * h * sizeof(float);
    size_t y_FileSize = n * k * sizeof(float);
    size_t expertIdx_FileSize = n * k * sizeof(int32_t);
    size_t out_FileSize = n * h * sizeof(float);
    size_t workspace_FileSize = 16781184;
    size_t tiling_FileSize = sizeof(MoeGatingTopKTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(bias_FileSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_FileSize);
    uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(out_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);
    MoeGatingTopKTilingData* tiling_data = reinterpret_cast<MoeGatingTopKTilingData*>(tiling);
    tiling_data->groupSelectMode = 1;
    tiling_data->renorm = 0;
    tiling_data->normType = 1;
    tiling_data->groupCount = 8;

    ICPU_SET_TILING_KEY(tilingKey);
    // ICPU_RUN_KF(moe_gating_top_k, blockDim, x, bias, y, expertIdx, out, workspace, (uint8_t*)tiling_data);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)expertIdx);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(moe_gating_top_k_test, test_case_1)
{
    size_t n = 16;
    size_t k = 8;
    size_t h = 256;
    uint64_t tilingKey = 1;
    uint32_t blockDim = 16;

    size_t x_FileSize = n * h * sizeof(float);
    size_t bias_FileSize = n * h * sizeof(float);
    size_t y_FileSize = n * k * sizeof(float);
    size_t expertIdx_FileSize = n * k * sizeof(int32_t);
    size_t out_FileSize = n * h * sizeof(float);
    size_t workspace_FileSize = 16781184;
    size_t tiling_FileSize = sizeof(MoeGatingTopKTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(bias_FileSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_FileSize);
    uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(out_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);
    MoeGatingTopKTilingData* tiling_data = reinterpret_cast<MoeGatingTopKTilingData*>(tiling);
    tiling_data->groupSelectMode = 1;
    tiling_data->renorm = 0;
    tiling_data->normType = 1;
    tiling_data->groupCount = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    // ICPU_RUN_KF(moe_gating_top_k, blockDim, x, bias, y, expertIdx, out, workspace, (uint8_t *)tiling_data);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)expertIdx);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(moe_gating_top_k_test, test_case_2)
{
    size_t n = 16;
    size_t k = 8;
    size_t h = 64;
    uint64_t tilingKey = 2;
    uint32_t blockDim = 16;

    size_t x_FileSize = n * h * sizeof(float);
    size_t bias_FileSize = n * h * sizeof(float);
    size_t y_FileSize = n * k * sizeof(float);
    size_t expertIdx_FileSize = n * k * sizeof(int32_t);
    size_t out_FileSize = n * h * sizeof(float);
    size_t workspace_FileSize = 16781184;
    size_t tiling_FileSize = sizeof(MoeGatingTopKTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(bias_FileSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_FileSize);
    uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(out_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);
    MoeGatingTopKTilingData* tiling_data = reinterpret_cast<MoeGatingTopKTilingData*>(tiling);
    tiling_data->groupSelectMode = 0;
    tiling_data->renorm = 0;
    tiling_data->normType = 0;
    tiling_data->groupCount = 8;

    ICPU_SET_TILING_KEY(tilingKey);
    // ICPU_RUN_KF(moe_gating_top_k, blockDim, x, bias, y, expertIdx, out, workspace, (uint8_t *)tiling_data);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)expertIdx);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)workspace);
}