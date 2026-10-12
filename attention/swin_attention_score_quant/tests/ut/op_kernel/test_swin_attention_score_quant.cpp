/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aglu.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_swin_attention_score_quant.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void swin_attention_score_quant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scale_quant, GM_ADDR scale_dequant1, GM_ADDR scale_dequant2,
    GM_ADDR bias_quant, GM_ADDR bias_dequant1, GM_ADDR bias_dequant2, GM_ADDR padding_mask1, GM_ADDR padding_mask2,
    GM_ADDR attention_score, GM_ADDR workspace, GM_ADDR tiling);

class swin_attention_score_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "swin_attention_score_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "swin_attention_score_quant_test TearDown\n" << endl;
    }
};

TEST_F(swin_attention_score_quant_test, test_case_has_mask_0001)
{
    size_t b = 128;
    size_t n = 12;
    size_t s = 49;
    size_t h = 32;
    size_t round16S = 64;
    size_t qkvByteSize = b * n * s * h * sizeof(int8_t);
    size_t scaleByteSize = 1 * round16S * sizeof(half);
    size_t scale1ByteSize = 1 * s * sizeof(u_int64_t);
    size_t scale2ByteSize = 1 * h * sizeof(u_int64_t);
    size_t biasByteSize = 1 * round16S * sizeof(half);
    size_t bias1ByteSize = 1 * s * sizeof(int32_t);
    size_t bias2ByteSize = 1 * h * sizeof(int32_t);
    size_t mask1ByteSize = 1 * n * s * s * sizeof(half);
    size_t mask2ByteSize = 1 * n * s * s * sizeof(half);
    size_t yByteSize = b * n * s * h * sizeof(half);
    size_t tiling_data_size = sizeof(SwinAttentionScoreQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1ByteSize);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2ByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(bias1ByteSize);
    uint8_t* bias2 = (uint8_t*)AscendC::GmAlloc(bias2ByteSize);
    uint8_t* mask1 = (uint8_t*)AscendC::GmAlloc(mask1ByteSize);
    uint8_t* mask2 = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SwinAttentionScoreQuantTilingData* tilingData = reinterpret_cast<SwinAttentionScoreQuantTilingData*>(tiling);
    tilingData->coreLoops = b;
    tilingData->dimB = b;
    tilingData->dimN = n;
    tilingData->dimS = s;
    tilingData->dimH = h;
    tilingData->qSize = 2048;
    tilingData->kSize = 2048;
    tilingData->pSize = 4096;
    tilingData->vSize = 2048;
    tilingData->cubeSharedUbSize = 4096;
    tilingData->vecSharedUbSize = 36864;
    tilingData->qkBmmTilingData.usedCoreNum = 1;
    tilingData->qkBmmTilingData.M = 64;
    tilingData->qkBmmTilingData.N = 64;
    tilingData->qkBmmTilingData.Ka = 32;
    tilingData->qkBmmTilingData.Kb = 32;
    tilingData->qkBmmTilingData.singleCoreM = 64;
    tilingData->qkBmmTilingData.singleCoreN = 49;
    tilingData->qkBmmTilingData.singleCoreK = 32;
    tilingData->qkBmmTilingData.baseM = 64;
    tilingData->qkBmmTilingData.baseN = 64;
    tilingData->qkBmmTilingData.baseK = 32;
    tilingData->qkBmmTilingData.depthA1 = 1;
    tilingData->qkBmmTilingData.depthB1 = 1;
    tilingData->qkBmmTilingData.stepM = 1;
    tilingData->qkBmmTilingData.stepN = 1;
    tilingData->qkBmmTilingData.isBias = 1;
    tilingData->qkBmmTilingData.transLength = 8192;
    tilingData->qkBmmTilingData.iterateOrder = 0;
    tilingData->qkBmmTilingData.shareMode = 0;
    tilingData->qkBmmTilingData.shareL1Size = 0;
    tilingData->qkBmmTilingData.shareL0CSize = 16384;
    tilingData->qkBmmTilingData.shareUbSize = 256;
    tilingData->qkBmmTilingData.batchM = 1;
    tilingData->qkBmmTilingData.batchN = 1;
    tilingData->qkBmmTilingData.singleBatchM = 1;
    tilingData->qkBmmTilingData.singleBatchN = 1;
    tilingData->qkBmmTilingData.stepKa = 1;
    tilingData->qkBmmTilingData.stepKb = 1;
    tilingData->qkBmmTilingData.dbL0A = 2;
    tilingData->qkBmmTilingData.dbL0B = 2;
    tilingData->qkBmmTilingData.dbL0C = 1;
    tilingData->pvBmmTilingData.usedCoreNum = 1;
    tilingData->pvBmmTilingData.M = 64;
    tilingData->pvBmmTilingData.N = 32;
    tilingData->pvBmmTilingData.Ka = 64;
    tilingData->pvBmmTilingData.Kb = 64;
    tilingData->pvBmmTilingData.singleCoreM = 64;
    tilingData->pvBmmTilingData.singleCoreN = 32;
    tilingData->pvBmmTilingData.singleCoreK = 49;
    tilingData->pvBmmTilingData.baseM = 64;
    tilingData->pvBmmTilingData.baseN = 32;
    tilingData->pvBmmTilingData.baseK = 64;
    tilingData->pvBmmTilingData.depthA1 = 1;
    tilingData->pvBmmTilingData.depthB1 = 1;
    tilingData->pvBmmTilingData.stepM = 1;
    tilingData->pvBmmTilingData.stepN = 1;
    tilingData->pvBmmTilingData.isBias = 1;
    tilingData->pvBmmTilingData.transLength = 4096;
    tilingData->pvBmmTilingData.iterateOrder = 0;
    tilingData->pvBmmTilingData.shareMode = 0;
    tilingData->pvBmmTilingData.shareL1Size = 0;
    tilingData->pvBmmTilingData.shareL0CSize = 16384;
    tilingData->pvBmmTilingData.shareUbSize = 4224;
    tilingData->pvBmmTilingData.batchM = 1;
    tilingData->pvBmmTilingData.batchN = 1;
    tilingData->pvBmmTilingData.singleBatchM = 1;
    tilingData->pvBmmTilingData.singleBatchN = 1;
    tilingData->pvBmmTilingData.stepKa = 1;
    tilingData->pvBmmTilingData.stepKb = 1;
    tilingData->pvBmmTilingData.dbL0A = 2;
    tilingData->pvBmmTilingData.dbL0B = 2;
    tilingData->pvBmmTilingData.dbL0C = 1;
    tilingData->softmaxTilingData.srcM = 64;
    tilingData->softmaxTilingData.srcK = 64;
    tilingData->softmaxTilingData.srcSize = 4096;
    tilingData->softmaxTilingData.outMaxM = 64;
    tilingData->softmaxTilingData.outMaxK = 16;
    tilingData->softmaxTilingData.outMaxSize = 1024;
    tilingData->softmaxTilingData.splitM = 64;
    tilingData->softmaxTilingData.splitK = 64;
    tilingData->softmaxTilingData.splitSize = 4096;
    tilingData->softmaxTilingData.reduceM = 64;
    tilingData->softmaxTilingData.reduceK = 16;
    tilingData->softmaxTilingData.reduceSize = 1024;
    tilingData->softmaxTilingData.rangeM = 1;
    tilingData->softmaxTilingData.tailM = 0;
    tilingData->softmaxTilingData.tailSplitSize = 0;
    tilingData->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(swin_attention_score_quant, blockDim, q, k, v, scale, scale1, scale2, bias, bias1, bias2, mask1, mask2,
                y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    AscendC::GmFree(scale);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(bias2);
    AscendC::GmFree(mask1);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swin_attention_score_quant_test, test_case_no_mask_0002)
{
    size_t b = 128;
    size_t n = 12;
    size_t s = 49;
    size_t h = 32;
    size_t round16S = 64;
    size_t qkvByteSize = b * n * s * h * sizeof(int8_t);
    size_t scaleByteSize = 1 * round16S * sizeof(half);
    size_t scale1ByteSize = 1 * s * sizeof(u_int64_t);
    size_t scale2ByteSize = 1 * h * sizeof(u_int64_t);
    size_t biasByteSize = 1 * round16S * sizeof(half);
    size_t bias1ByteSize = 1 * s * sizeof(int32_t);
    size_t bias2ByteSize = 1 * h * sizeof(int32_t);
    size_t mask1ByteSize = 1 * n * s * s * sizeof(half);
    size_t mask2ByteSize = 1 * n * s * s * sizeof(half);
    size_t yByteSize = b * n * s * h * sizeof(half);
    size_t tiling_data_size = sizeof(SwinAttentionScoreQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(qkvByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1ByteSize);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2ByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(bias1ByteSize);
    uint8_t* bias2 = (uint8_t*)AscendC::GmAlloc(bias2ByteSize);
    uint8_t* mask1 = nullptr;
    uint8_t* mask2 = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SwinAttentionScoreQuantTilingData* tilingData = reinterpret_cast<SwinAttentionScoreQuantTilingData*>(tiling);
    tilingData->coreLoops = b * n;
    tilingData->dimB = b;
    tilingData->dimN = n;
    tilingData->dimS = s;
    tilingData->dimH = h;
    tilingData->qSize = 2048;
    tilingData->kSize = 2048;
    tilingData->pSize = 4096;
    tilingData->vSize = 2048;
    tilingData->cubeSharedUbSize = 4096;
    tilingData->vecSharedUbSize = 36864;
    tilingData->qkBmmTilingData.usedCoreNum = 1;
    tilingData->qkBmmTilingData.M = 64;
    tilingData->qkBmmTilingData.N = 64;
    tilingData->qkBmmTilingData.Ka = 32;
    tilingData->qkBmmTilingData.Kb = 32;
    tilingData->qkBmmTilingData.singleCoreM = 64;
    tilingData->qkBmmTilingData.singleCoreN = 49;
    tilingData->qkBmmTilingData.singleCoreK = 32;
    tilingData->qkBmmTilingData.baseM = 64;
    tilingData->qkBmmTilingData.baseN = 64;
    tilingData->qkBmmTilingData.baseK = 32;
    tilingData->qkBmmTilingData.depthA1 = 1;
    tilingData->qkBmmTilingData.depthB1 = 1;
    tilingData->qkBmmTilingData.stepM = 1;
    tilingData->qkBmmTilingData.stepN = 1;
    tilingData->qkBmmTilingData.isBias = 1;
    tilingData->qkBmmTilingData.transLength = 8192;
    tilingData->qkBmmTilingData.iterateOrder = 0;
    tilingData->qkBmmTilingData.shareMode = 0;
    tilingData->qkBmmTilingData.shareL1Size = 0;
    tilingData->qkBmmTilingData.shareL0CSize = 16384;
    tilingData->qkBmmTilingData.shareUbSize = 256;
    tilingData->qkBmmTilingData.batchM = 1;
    tilingData->qkBmmTilingData.batchN = 1;
    tilingData->qkBmmTilingData.singleBatchM = 1;
    tilingData->qkBmmTilingData.singleBatchN = 1;
    tilingData->qkBmmTilingData.stepKa = 1;
    tilingData->qkBmmTilingData.stepKb = 1;
    tilingData->qkBmmTilingData.dbL0A = 2;
    tilingData->qkBmmTilingData.dbL0B = 2;
    tilingData->qkBmmTilingData.dbL0C = 1;
    tilingData->pvBmmTilingData.usedCoreNum = 1;
    tilingData->pvBmmTilingData.M = 64;
    tilingData->pvBmmTilingData.N = 32;
    tilingData->pvBmmTilingData.Ka = 64;
    tilingData->pvBmmTilingData.Kb = 64;
    tilingData->pvBmmTilingData.singleCoreM = 64;
    tilingData->pvBmmTilingData.singleCoreN = 32;
    tilingData->pvBmmTilingData.singleCoreK = 49;
    tilingData->pvBmmTilingData.baseM = 64;
    tilingData->pvBmmTilingData.baseN = 32;
    tilingData->pvBmmTilingData.baseK = 64;
    tilingData->pvBmmTilingData.depthA1 = 1;
    tilingData->pvBmmTilingData.depthB1 = 1;
    tilingData->pvBmmTilingData.stepM = 1;
    tilingData->pvBmmTilingData.stepN = 1;
    tilingData->pvBmmTilingData.isBias = 1;
    tilingData->pvBmmTilingData.transLength = 4096;
    tilingData->pvBmmTilingData.iterateOrder = 0;
    tilingData->pvBmmTilingData.shareMode = 0;
    tilingData->pvBmmTilingData.shareL1Size = 0;
    tilingData->pvBmmTilingData.shareL0CSize = 16384;
    tilingData->pvBmmTilingData.shareUbSize = 4224;
    tilingData->pvBmmTilingData.batchM = 1;
    tilingData->pvBmmTilingData.batchN = 1;
    tilingData->pvBmmTilingData.singleBatchM = 1;
    tilingData->pvBmmTilingData.singleBatchN = 1;
    tilingData->pvBmmTilingData.stepKa = 1;
    tilingData->pvBmmTilingData.stepKb = 1;
    tilingData->pvBmmTilingData.dbL0A = 2;
    tilingData->pvBmmTilingData.dbL0B = 2;
    tilingData->pvBmmTilingData.dbL0C = 1;
    tilingData->softmaxTilingData.srcM = 64;
    tilingData->softmaxTilingData.srcK = 64;
    tilingData->softmaxTilingData.srcSize = 4096;
    tilingData->softmaxTilingData.outMaxM = 64;
    tilingData->softmaxTilingData.outMaxK = 16;
    tilingData->softmaxTilingData.outMaxSize = 1024;
    tilingData->softmaxTilingData.splitM = 64;
    tilingData->softmaxTilingData.splitK = 64;
    tilingData->softmaxTilingData.splitSize = 4096;
    tilingData->softmaxTilingData.reduceM = 64;
    tilingData->softmaxTilingData.reduceK = 16;
    tilingData->softmaxTilingData.reduceSize = 1024;
    tilingData->softmaxTilingData.rangeM = 1;
    tilingData->softmaxTilingData.tailM = 0;
    tilingData->softmaxTilingData.tailSplitSize = 0;
    tilingData->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(swin_attention_score_quant, blockDim, q, k, v, scale, scale1, scale2, bias, bias1, bias2, mask1, mask2,
                y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    AscendC::GmFree(scale);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(bias2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
