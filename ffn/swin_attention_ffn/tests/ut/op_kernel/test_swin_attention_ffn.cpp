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
 #include "data_utils.h"
 #include "tiling_case_executor.h"
 #include "swin_attention_ffn_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void swin_attention_ffn(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, 
    GM_ADDR tiling);

class swin_attention_ffn_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "swin_attention_ffn_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "swin_attention_ffn_test TearDown\n" << endl;
    }
};

TEST_F(swin_attention_ffn_test, test_case0)
{
    size_t x1ByteSize = 1 * 64 * 128 * sizeof(half);
    size_t x2ByteSize = 128 * 128 * sizeof(half);
    size_t biasByteSize = 128 * sizeof(half);
    size_t x3ByteSize = 32 * 32 * 64 * sizeof(half);
    size_t yByteSize = 1 * 64 * 128 * sizeof(half);
    size_t tiling_data_size = sizeof(SwinAttentionFFNTilingData);
    uint32_t blockDim = 24;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(x3ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SwinAttentionFFNTilingData* tilingDatafromBin =
        reinterpret_cast<SwinAttentionFFNTilingData*>(tiling);

    TCubeTiling bmmTilingData;

    tilingDatafromBin->batchSize = 0;
    tilingDatafromBin->bmmFormerNum = 0;
    tilingDatafromBin->bmmTailNum = 0;
    tilingDatafromBin->bmmFormerBatchNum = 1;
    tilingDatafromBin->bmmTailBatchNum = 0;
    tilingDatafromBin->aivNum = 0;
    tilingDatafromBin->shift1 = 4;
    tilingDatafromBin->shift2 = 4;
    tilingDatafromBin->tpBlockSize = 1024;
    tilingDatafromBin->tpSpaceCnt = 1;
    tilingDatafromBin->tpSpaceH = 1;
    tilingDatafromBin->tpSpaceW = 8;
    tilingDatafromBin->blockInSpace = 8;
    tilingDatafromBin->tpSpaceSize = 8192;
    tilingDatafromBin->tpSpaceWTransposed = 1;
    tilingDatafromBin->tpSpaceHTransposed = 8;

    tilingDatafromBin->bmmTilingData.usedCoreNum = 1;
    tilingDatafromBin->bmmTilingData.M = 512;
    tilingDatafromBin->bmmTilingData.N = 128;
    tilingDatafromBin->bmmTilingData.Ka = 128;
    tilingDatafromBin->bmmTilingData.Kb = 128;
    tilingDatafromBin->bmmTilingData.singleCoreM = 512;
    tilingDatafromBin->bmmTilingData.singleCoreN = 128;
    tilingDatafromBin->bmmTilingData.singleCoreK = 128;
    tilingDatafromBin->bmmTilingData.baseM = 128;
    tilingDatafromBin->bmmTilingData.baseN = 128;
    tilingDatafromBin->bmmTilingData.baseK = 128;
    tilingDatafromBin->bmmTilingData.depthA1 = 4;
    tilingDatafromBin->bmmTilingData.depthB1 = 1;
    tilingDatafromBin->bmmTilingData.stepM = 4;
    tilingDatafromBin->bmmTilingData.stepN = 1;
    tilingDatafromBin->bmmTilingData.isBias = 1;
    tilingDatafromBin->bmmTilingData.transLength = 0;
    tilingDatafromBin->bmmTilingData.iterateOrder = 1;
    tilingDatafromBin->bmmTilingData.shareMode = 0;
    tilingDatafromBin->bmmTilingData.shareL1Size = 164096;
    tilingDatafromBin->bmmTilingData.shareL0CSize = 65536;
    tilingDatafromBin->bmmTilingData.shareUbSize = 0;
    tilingDatafromBin->bmmTilingData.batchM = 1;
    tilingDatafromBin->bmmTilingData.batchN = 1;
    tilingDatafromBin->bmmTilingData.singleBatchM = 1;
    tilingDatafromBin->bmmTilingData.singleBatchN = 1;
    tilingDatafromBin->bmmTilingData.stepKa = 1;
    tilingDatafromBin->bmmTilingData.stepKb = 1;
    tilingDatafromBin->bmmTilingData.depthAL1CacheUB = 0;
    tilingDatafromBin->bmmTilingData.depthBL1CacheUB = 0;
    tilingDatafromBin->bmmTilingData.dbL0A = 2;
    tilingDatafromBin->bmmTilingData.dbL0B = 2;
    tilingDatafromBin->bmmTilingData.dbL0C = 1;
    tilingDatafromBin->bmmTilingData.ALayoutInfoB = 0;
    tilingDatafromBin->bmmTilingData.ALayoutInfoS = 0;
    tilingDatafromBin->bmmTilingData.ALayoutInfoN = 0;
    tilingDatafromBin->bmmTilingData.ALayoutInfoG = 0;
    tilingDatafromBin->bmmTilingData.ALayoutInfoD = 0;
    tilingDatafromBin->bmmTilingData.BLayoutInfoB = 0;
    tilingDatafromBin->bmmTilingData.BLayoutInfoS = 0;
    tilingDatafromBin->bmmTilingData.BLayoutInfoN = 0;
    tilingDatafromBin->bmmTilingData.BLayoutInfoG = 0;
    tilingDatafromBin->bmmTilingData.BLayoutInfoD = 0;
    tilingDatafromBin->bmmTilingData.CLayoutInfoB = 0;
    tilingDatafromBin->bmmTilingData.CLayoutInfoS1 = 0;
    tilingDatafromBin->bmmTilingData.CLayoutInfoN = 0;
    tilingDatafromBin->bmmTilingData.CLayoutInfoG = 0;
    tilingDatafromBin->bmmTilingData.CLayoutInfoS2 = 0;
    tilingDatafromBin->bmmTilingData.BatchNum = 0;
    tilingDatafromBin->bmmTilingData.mxTypePara = 0;


    tilingDatafromBin->dataNumPerBatchA = 65536;
    tilingDatafromBin->dataNumPerBatchD = 65536;
    tilingDatafromBin->dataNumPerLoop = 16384;
    tilingDatafromBin->reserved = 0;


    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        swin_attention_ffn, blockDim, x1, x2, bias, x3, y, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(bias);
    AscendC::GmFree(x3);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}