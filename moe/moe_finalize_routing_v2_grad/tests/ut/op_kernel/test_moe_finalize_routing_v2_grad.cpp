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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_moe_finalize_routing_v2_grad_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2_grad(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias,
    GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace, GM_ADDR tiling);

class moe_finalize_routing_v2_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_finalize_routing_v2_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_finalize_routing_v2_grad_test TearDown\n" << endl;
    }
};

TEST_F(moe_finalize_routing_v2_grad_test, test_case_10001_001)
{
    size_t gradYByteSize = 5 * 8 * sizeof(float);
    size_t expandedRowIdxByteSize = 5 * sizeof(int32_t);
    size_t gradExpandedXByteSize = 5 * 8 * sizeof(float);
    size_t gradScalesByteSize = 5 * 1 * sizeof(float);

    size_t tilingDataByteSize = sizeof(MoeFinalizeRoutingV2GradTilingData);
    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdxByteSize);
    uint8_t* gradExpandedX = (uint8_t*)AscendC::GmAlloc(gradExpandedXByteSize);
    uint8_t* gradScales = (uint8_t*)AscendC::GmAlloc(gradScalesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataByteSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeFinalizeRoutingV2GradTilingData* tilingData = reinterpret_cast<MoeFinalizeRoutingV2GradTilingData*>(tiling);

    tilingData->initOutNeedCoreNum = 5;
    tilingData->initOutEachCoreBatchNum = 0;
    tilingData->initOutModCoreNum = 5;
    tilingData->computeNeedCoreNum = 5;
    tilingData->computeEachCoreBatchNum = 0;
    tilingData->computeModCoreNum = 5;
    tilingData->dropPadMode = 0;
    tilingData->topK = 1;
    tilingData->hidden = 8;
    tilingData->expandedXDim0 = 5;
    tilingData->hiddenPrePart = 49080;
    tilingData->hiddenInnerLoops = 0;
    tilingData->hiddenLastPart = 0;
    tilingData->tilingKey = 10001;

    ICPU_SET_TILING_KEY(10001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        moe_finalize_routing_v2_grad, blockDim, gradY, expandedRowIdx, nullptr, nullptr, nullptr, nullptr,
        gradExpandedX, gradScales, workspace, tiling);

    AscendC::GmFree(gradY);
    AscendC::GmFree(expandedRowIdx);
    AscendC::GmFree(gradExpandedX);
    AscendC::GmFree(gradScales);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_finalize_routing_v2_grad_test, test_case_10002_001)
{
    size_t gradYByteSize = 5 * 262144 * sizeof(float);
    size_t expandedRowIdxByteSize = 5 * sizeof(int32_t);
    size_t gradExpandedXByteSize = 5 * 262144 * sizeof(float);
    size_t gradScalesByteSize = 5 * 1 * sizeof(float);

    size_t tilingDataByteSize = sizeof(MoeFinalizeRoutingV2GradTilingData);
    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdxByteSize);
    uint8_t* gradExpandedX = (uint8_t*)AscendC::GmAlloc(gradExpandedXByteSize);
    uint8_t* gradScales = (uint8_t*)AscendC::GmAlloc(gradScalesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataByteSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeFinalizeRoutingV2GradTilingData* tilingData = reinterpret_cast<MoeFinalizeRoutingV2GradTilingData*>(tiling);

    tilingData->initOutNeedCoreNum = 5;
    tilingData->initOutEachCoreBatchNum = 0;
    tilingData->initOutModCoreNum = 5;
    tilingData->computeNeedCoreNum = 5;
    tilingData->computeEachCoreBatchNum = 0;
    tilingData->computeModCoreNum = 5;
    tilingData->dropPadMode = 0;
    tilingData->topK = 1;
    tilingData->hidden = 262144;
    tilingData->expandedXDim0 = 5;
    tilingData->hiddenPrePart = 49080;
    tilingData->hiddenInnerLoops = 5;
    tilingData->hiddenLastPart = 16744;
    tilingData->tilingKey = 10002;

    ICPU_SET_TILING_KEY(10002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        moe_finalize_routing_v2_grad, blockDim, gradY, expandedRowIdx, nullptr, nullptr, nullptr, nullptr,
        gradExpandedX, gradScales, workspace, tiling);

    AscendC::GmFree(gradY);
    AscendC::GmFree(expandedRowIdx);
    AscendC::GmFree(gradExpandedX);
    AscendC::GmFree(gradScales);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_finalize_routing_v2_grad_test, test_case_20001_001)
{
    size_t gradYByteSize = 5 * 8 * sizeof(float);
    size_t expandedRowIdxByteSize = 5 * 3 * sizeof(int32_t);
    size_t expandedXByteSize = 5 * 3 * 8 * sizeof(float);
    size_t scalesByteSize = 5 * 3 * sizeof(float);
    size_t expertIdxByteSize = 5 * 3 * sizeof(int32_t);
    size_t biasByteSize = 8 * 8 * sizeof(float);
    size_t gradExpandedXByteSize = 5 * 3 * 8 * sizeof(float);
    size_t gradScalesByteSize = 5 * 3 * sizeof(float);

    size_t tilingDataByteSize = sizeof(MoeFinalizeRoutingV2GradTilingData);
    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdxByteSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedXByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdxByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* gradExpandedX = (uint8_t*)AscendC::GmAlloc(gradExpandedXByteSize);
    uint8_t* gradScales = (uint8_t*)AscendC::GmAlloc(gradScalesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataByteSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeFinalizeRoutingV2GradTilingData* tilingData = reinterpret_cast<MoeFinalizeRoutingV2GradTilingData*>(tiling);

    tilingData->initOutNeedCoreNum = 5;
    tilingData->initOutEachCoreBatchNum = 0;
    tilingData->initOutModCoreNum = 5;
    tilingData->computeNeedCoreNum = 5;
    tilingData->computeEachCoreBatchNum = 0;
    tilingData->computeModCoreNum = 5;
    tilingData->dropPadMode = 0;
    tilingData->topK = 3;
    tilingData->hidden = 8;
    tilingData->expandedXDim0 = 5;
    tilingData->hiddenPrePart = 9808;
    tilingData->hiddenInnerLoops = 0;
    tilingData->hiddenLastPart = 0;
    tilingData->tilingKey = 20001;

    ICPU_SET_TILING_KEY(20001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        moe_finalize_routing_v2_grad, blockDim, gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
        gradExpandedX, gradScales, workspace, tiling);

    AscendC::GmFree(gradY);
    AscendC::GmFree(expandedRowIdx);
    AscendC::GmFree(expandedX);
    AscendC::GmFree(scales);
    AscendC::GmFree(expertIdx);
    AscendC::GmFree(bias);
    AscendC::GmFree(gradExpandedX);
    AscendC::GmFree(gradScales);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_finalize_routing_v2_grad_test, test_case_20002_001)
{
    size_t gradYByteSize = 5 * 262144 * sizeof(float);
    size_t expandedRowIdxByteSize = 5 * 3 * sizeof(int32_t);
    size_t expandedXByteSize = 5 * 3 * 262144 * sizeof(float);
    size_t scalesByteSize = 5 * 3 * sizeof(float);
    size_t expertIdxByteSize = 5 * 3 * sizeof(int32_t);
    size_t biasByteSize = 8 * 262144 * sizeof(float);
    size_t gradExpandedXByteSize = 5 * 3 * 262144 * sizeof(float);
    size_t gradScalesByteSize = 5 * 3 * sizeof(float);

    size_t tilingDataByteSize = sizeof(MoeFinalizeRoutingV2GradTilingData);
    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdxByteSize);
    uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedXByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdxByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* gradExpandedX = (uint8_t*)AscendC::GmAlloc(gradExpandedXByteSize);
    uint8_t* gradScales = (uint8_t*)AscendC::GmAlloc(gradScalesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataByteSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeFinalizeRoutingV2GradTilingData* tilingData = reinterpret_cast<MoeFinalizeRoutingV2GradTilingData*>(tiling);

    tilingData->initOutNeedCoreNum = 5;
    tilingData->initOutEachCoreBatchNum = 0;
    tilingData->initOutModCoreNum = 5;
    tilingData->computeNeedCoreNum = 5;
    tilingData->computeEachCoreBatchNum = 0;
    tilingData->computeModCoreNum = 5;
    tilingData->dropPadMode = 0;
    tilingData->topK = 3;
    tilingData->hidden = 2621;
    tilingData->expandedXDim0 = 5;
    tilingData->hiddenPrePart = 9808;
    tilingData->hiddenInnerLoops = 26;
    tilingData->hiddenLastPart = 7136;
    tilingData->tilingKey = 20002;
    tilingData->hiddenPrePart = 8000;

    ICPU_SET_TILING_KEY(20001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        moe_finalize_routing_v2_grad, blockDim, gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
        gradExpandedX, gradScales, workspace, tiling);

    AscendC::GmFree(gradY);
    AscendC::GmFree(expandedRowIdx);
    AscendC::GmFree(expandedX);
    AscendC::GmFree(scales);
    AscendC::GmFree(expertIdx);
    AscendC::GmFree(bias);
    AscendC::GmFree(gradExpandedX);
    AscendC::GmFree(gradScales);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}