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
 #include "swin_transformer_ln_qkv_tiling.h"

using namespace std;
extern "C" __global__ __aicore__ void swin_transformer_ln_qkv(GM_ADDR inputX, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR weight, GM_ADDR bias, GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output, GM_ADDR workspace,
    GM_ADDR tiling);

class swin_transformer_ln_qkv_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "swin_transformer_ln_qkv_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "swin_transformer_ln_qkv_test TearDown\n" << endl;
    }
};

TEST_F(swin_transformer_ln_qkv_test, test_case0)
{
    size_t inputX_bytesize = 1 * 64 * 128 * sizeof(half);
    size_t gamma_bytesize = 128 * sizeof(half);
    size_t beta_bytesize = 128 * sizeof(half);
    size_t weight_bytesize = 128 * 384 * sizeof(half);
    size_t bias_bytesize = 384 * sizeof(half);

    size_t query_output_bytesize = 1 * 4 * 64 * 32 * sizeof(half);
    size_t key_output_bytesize = 1 * 4 * 64 * 32 * sizeof(half);
    size_t value_output_bytesize = 1 * 4 * 64 * 32 * sizeof(half);

    size_t tiling_data_size = sizeof(SwinTransformerLnQKVTilingData);//280
    uint32_t blockDim = 24;

    uint8_t* inputX = (uint8_t*)AscendC::GmAlloc(inputX_bytesize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_bytesize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_bytesize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weight_bytesize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(bias_bytesize);
    uint8_t* query_output = (uint8_t*)AscendC::GmAlloc(query_output_bytesize);
    uint8_t* key_output = (uint8_t*)AscendC::GmAlloc(key_output_bytesize);
    uint8_t* value_output = (uint8_t*)AscendC::GmAlloc(value_output_bytesize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1258291200);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SwinTransformerLnQKVTilingData* tilingDatafromBin =
        reinterpret_cast<SwinTransformerLnQKVTilingData*>(tiling);

    TCubeTiling mmTilingParams;
    SwinTransformerLnQKVBaseInfo opBaseInfo;
    SwinTransformerLnQKVLayernormTilingData layernormTilingParams;

    tilingDatafromBin->maxCoreNum = 0;
    tilingDatafromBin->useVectorNum = 1;
    tilingDatafromBin->workspaceSize = 1258291200;
    tilingDatafromBin->inputSizeSum = 8192;

    tilingDatafromBin->opBaseInfo.inputsize = 8192;
    tilingDatafromBin->opBaseInfo.hSize = 256;
    tilingDatafromBin->opBaseInfo.baseLoopNum = 0;
    tilingDatafromBin->opBaseInfo.remainderBlockNum = 0;
    tilingDatafromBin->layernormTilingParams.bLength = 8;
    tilingDatafromBin->layernormTilingParams.sLength = 256;
    tilingDatafromBin->layernormTilingParams.hLength = 128;
    tilingDatafromBin->layernormTilingParams.bsLength = 2048;
    tilingDatafromBin->layernormTilingParams.shLength = 32768;
    tilingDatafromBin->layernormTilingParams.loopSize = 262144;
    tilingDatafromBin->layernormTilingParams.elementPerBlock = 0;
    tilingDatafromBin->layernormTilingParams.remainderElementPerBlock = 262144;
    tilingDatafromBin->layernormTilingParams.innerLoopLength = 32768;
    tilingDatafromBin->layernormTilingParams.innerLoopNum = 8;
    tilingDatafromBin->layernormTilingParams.normalBlockElementOffset = 0;
    tilingDatafromBin->layernormTilingParams.rollOffset = 0;
    tilingDatafromBin->mmTilingParams.usedCoreNum = 1;
    tilingDatafromBin->mmTilingParams.M = 2048;
    tilingDatafromBin->mmTilingParams.N = 384;
    tilingDatafromBin->mmTilingParams.Ka = 128;
    tilingDatafromBin->mmTilingParams.Kb = 128;
    tilingDatafromBin->mmTilingParams.singleCoreM = 2048;
    tilingDatafromBin->mmTilingParams.singleCoreN = 384;
    tilingDatafromBin->mmTilingParams.singleCoreK = 128;
    tilingDatafromBin->mmTilingParams.baseM = 128;
    tilingDatafromBin->mmTilingParams.baseN = 256;
    tilingDatafromBin->mmTilingParams.baseK = 64;
    tilingDatafromBin->mmTilingParams.depthA1 = 16;
    tilingDatafromBin->mmTilingParams.depthB1 = 4;
    tilingDatafromBin->mmTilingParams.stepM = 4;
    tilingDatafromBin->mmTilingParams.stepN = 2;
    tilingDatafromBin->mmTilingParams.isBias = 1;
    tilingDatafromBin->mmTilingParams.transLength = 0;
    tilingDatafromBin->mmTilingParams.iterateOrder = 1;
    tilingDatafromBin->mmTilingParams.shareMode = 0;
    tilingDatafromBin->mmTilingParams.shareL1Size = 393728;
    tilingDatafromBin->mmTilingParams.shareL0CSize = 131072;
    tilingDatafromBin->mmTilingParams.shareUbSize = 0;
    tilingDatafromBin->mmTilingParams.batchM = 1;
    tilingDatafromBin->mmTilingParams.batchN = 1;
    tilingDatafromBin->mmTilingParams.singleBatchM = 1;
    tilingDatafromBin->mmTilingParams.singleBatchN = 1;
    tilingDatafromBin->mmTilingParams.stepKa = 2;
    tilingDatafromBin->mmTilingParams.stepKb = 2;
    tilingDatafromBin->mmTilingParams.depthAL1CacheUB = 0;
    tilingDatafromBin->mmTilingParams.depthBL1CacheUB = 0;
    tilingDatafromBin->mmTilingParams.dbL0A = 2;
    tilingDatafromBin->mmTilingParams.dbL0B = 2;
    tilingDatafromBin->mmTilingParams.dbL0C = 1;
    tilingDatafromBin->mmTilingParams.ALayoutInfoB = 0;
    tilingDatafromBin->mmTilingParams.ALayoutInfoS = 0;
    tilingDatafromBin->mmTilingParams.ALayoutInfoN = 0;
    tilingDatafromBin->mmTilingParams.ALayoutInfoG = 0;
    tilingDatafromBin->mmTilingParams.ALayoutInfoD = 0;
    tilingDatafromBin->mmTilingParams.BLayoutInfoB = 0;
    tilingDatafromBin->mmTilingParams.BLayoutInfoS = 0;
    tilingDatafromBin->mmTilingParams.BLayoutInfoN = 0;
    tilingDatafromBin->mmTilingParams.BLayoutInfoG = 0;
    tilingDatafromBin->mmTilingParams.BLayoutInfoD = 0;
    tilingDatafromBin->mmTilingParams.CLayoutInfoB = 0;
    tilingDatafromBin->mmTilingParams.CLayoutInfoS1 = 0;
    tilingDatafromBin->mmTilingParams.CLayoutInfoN = 0;
    tilingDatafromBin->mmTilingParams.CLayoutInfoG = 0;
    tilingDatafromBin->mmTilingParams.CLayoutInfoS2 = 0;
    tilingDatafromBin->mmTilingParams.BatchNum = 0;
    tilingDatafromBin->mmTilingParams.mxTypePara = 0;


    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        swin_transformer_ln_qkv, blockDim, inputX, gamma, beta, weight, bias, query_output, key_output, value_output, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(inputX);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(query_output);
    AscendC::GmFree(key_output);
    AscendC::GmFree(value_output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

}