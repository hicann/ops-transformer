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
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "moe_fused_topk_tiling.h"

using namespace std;

extern "C" void moe_fused_topk(GM_ADDR x, GM_ADDR addNum, GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y,
                               GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

struct MoeFusedTopkTestParam {
    string caseName;

    size_t dataTypeSize1;
    size_t dataTypeSize2;

    uint64_t tilingKey;
    MoeFusedTopkTilingData tiling;
};

class MoeFusedTopkTest : public testing::TestWithParam<MoeFusedTopkTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFusedTopkTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFusedTopkTest TearDown" << std::endl;
    }
};

TEST_P(MoeFusedTopkTest, test_case_moe_fused_topk)
{
    MoeFusedTopkTestParam param = GetParam();

    uint32_t tilingKey = param.tilingKey;
    MoeFusedTopkTilingData tilingParam = param.tiling;

    int64_t xShapeSize = tilingParam.firstDimSize * tilingParam.secondDimSize;
    int64_t addNumShapeSize = tilingParam.addNumDimSize;

    int64_t mappingNumShapeSize = tilingParam.expertNum;
    int64_t mappingTableShapeSize = tilingParam.expertNum * tilingParam.tableDim;

    int64_t yShapeSize = tilingParam.firstDimSize * tilingParam.topK;
    int64_t indicesShapeSize = tilingParam.firstDimSize * tilingParam.topK;

    int64_t xByteSize = xShapeSize * param.dataTypeSize1;
    int64_t addNumByteSize = addNumShapeSize * param.dataTypeSize1;

    int64_t mappingNumByteSize = mappingNumShapeSize * param.dataTypeSize2;
    int64_t mappingTableByteSize = mappingTableShapeSize * param.dataTypeSize2;

    int64_t yByteSize = yShapeSize * param.dataTypeSize2;
    int64_t indicesByteSize = indicesShapeSize * param.dataTypeSize2;

    int64_t workspaceSize = 16 * 1024 * 1024 + tilingParam.blockNum * tilingParam.workspacePerCore;
    int64_t tilingSize = sizeof(MoeFusedTopkTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xByteSize);
    uint8_t *addNum = (uint8_t *)AscendC::GmAlloc(addNumByteSize);
    uint8_t *mappingNum = (uint8_t *)AscendC::GmAlloc(mappingNumByteSize);
    uint8_t *mappingTable = (uint8_t *)AscendC::GmAlloc(mappingTableByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yByteSize);
    uint8_t *indices = (uint8_t *)AscendC::GmAlloc(indicesByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    MoeFusedTopkTilingData *tilingData = reinterpret_cast<MoeFusedTopkTilingData *>(tiling);
    tilingData->firstDimSize = tilingParam.firstDimSize;
    tilingData->secondDimSize = tilingParam.secondDimSize;
    tilingData->addNumDimSize = tilingParam.addNumDimSize;
    tilingData->groupNum = tilingParam.groupNum;
    tilingData->groupTopk = tilingParam.groupTopk;
    tilingData->topN = tilingParam.topN;
    tilingData->topK = tilingParam.topK;

    tilingData->activateType = tilingParam.activateType;
    tilingData->isNorm = tilingParam.isNorm;
    tilingData->scale = tilingParam.scale;
    tilingData->groupEles = tilingParam.groupEles;
    tilingData->blockNum = tilingParam.blockNum;
    tilingData->ubFactorElement = tilingParam.ubFactorElement;
    tilingData->batchPerCore = tilingParam.batchPerCore;
    tilingData->tailBatch = tilingParam.tailBatch;

    tilingData->expertNum = tilingParam.expertNum;
    tilingData->tableDim = tilingParam.tableDim;
    tilingData->topkMaxValue = tilingParam.topkMaxValue;
    tilingData->topkMinValue = tilingParam.topkMinValue;
    tilingData->reserved = tilingParam.reserved;
    tilingData->workspacePerCore = tilingParam.workspacePerCore;
    tilingData->topkTilingData = tilingParam.topkTilingData;

    uint32_t blockDim = tilingParam.blockNum;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(moe_fused_topk, blockDim, x, addNum, mappingNum, mappingTable, y, indices, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)addNum);
    AscendC::GmFree((void *)mappingNum);
    AscendC::GmFree((void *)mappingTable);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)indices);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

static MoeFusedTopkTestParam cases[] = {
    {"test_case_float32",
     sizeof(float),
     sizeof(int32_t),
     0,
     {128,  128,
      128,  8,
      4,    4,
      8,    0,
      1,    1.0,
      16,   40,
      5556, 3,
      8,    128,
      128,  1024,
      1024, 0,
      512,  {160, 256, 64, 1, 8, 8, 16, 2, 4, 6, 2, 64, 1, 128, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
};

INSTANTIATE_TEST_CASE_P(MoeFusedTopk, MoeFusedTopkTest, testing::ValuesIn(cases));
