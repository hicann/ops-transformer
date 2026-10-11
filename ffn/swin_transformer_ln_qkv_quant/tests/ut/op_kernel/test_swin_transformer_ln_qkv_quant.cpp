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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_swin_transformer_ln_qkv_quant.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void swin_transformer_ln_qkv_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,
                                                                    GM_ADDR weight, GM_ADDR bias, GM_ADDR quant_scale,
                                                                    GM_ADDR quant_offset, GM_ADDR dequant_scale,
                                                                    GM_ADDR query_output, GM_ADDR key_output,
                                                                    GM_ADDR value_output, GM_ADDR workspace,
                                                                    GM_ADDR tiling);

class swin_transformer_ln_qkv_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "swin_transformer_ln_qkv_quant_fast_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "swin_transformer_ln_qkv_quant_fast_test TearDown\n" << endl;
    }
};

TEST_F(swin_transformer_ln_qkv_quant_test, test_case_fp16_swin_qkv_quant_fast_op)
{
    size_t b = 1;
    size_t s = 49;
    size_t h = 32;

    size_t tiling_data_size = sizeof(SwinTransformerLnQkvQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(b * s * h * sizeof(uint16_t));
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(h * sizeof(uint16_t));
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(h * sizeof(uint16_t));
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(h * 3 * h);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(3 * h * sizeof(uint32_t));
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(h * sizeof(uint16_t));
    uint8_t* quant_offset = (uint8_t*)AscendC::GmAlloc(h * sizeof(uint16_t));
    uint8_t* dequant_scale = (uint8_t*)AscendC::GmAlloc(3 * h * sizeof(uint64_t));
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(10 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(b * s * h * sizeof(uint16_t));
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(b * s * h * sizeof(uint16_t));
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(b * s * h * sizeof(uint16_t));

    char* path_ = get_current_dir_name();
    string path(path_);

    SwinTransformerLnQkvQuantTilingData* tilingDatafromBin =
        reinterpret_cast<SwinTransformerLnQkvQuantTilingData*>(tiling);

    tilingDatafromBin->opBaseInfo.bSize = 1;
    tilingDatafromBin->opBaseInfo.sSize = 49;
    tilingDatafromBin->opBaseInfo.hSize = 32;

    tilingDatafromBin->weightN = 96;
    tilingDatafromBin->size = 1 * 49 * 32;
    tilingDatafromBin->opBaseInfo.lnBaseM = 7;
    tilingDatafromBin->opBaseInfo.lnBaseK = 32;
    tilingDatafromBin->opBaseInfo.hWinSize = 7;
    tilingDatafromBin->opBaseInfo.wWinSize = 7;
    tilingDatafromBin->opBaseInfo.sizePerHead = 32;
    tilingDatafromBin->opBaseInfo.lnBufferM = 16;
    tilingDatafromBin->opBaseInfo.lnBufferK = 32;
    tilingDatafromBin->opBaseInfo.lnKSubLoop = 1;

    tilingDatafromBin->mmInfo.mmSizeN = 96;
    tilingDatafromBin->mmInfo.mmSizeM = 7;
    tilingDatafromBin->mmInfo.mmSizeK = 32;
    tilingDatafromBin->opBaseInfo.lnMSubLoop = 1;
    tilingDatafromBin->opBaseInfo.headNum = 1;
    tilingDatafromBin->opBaseInfo.patchHeight = 1;
    tilingDatafromBin->opBaseInfo.patchWeight = 1;
    tilingDatafromBin->opBaseInfo.lnBufferM = 16;
    tilingDatafromBin->opBaseInfo.lnBufferK = 32;
    tilingDatafromBin->tmpBufferForQuant = 512;
    tilingDatafromBin->tmpShareBufferForLn = 80 * 1024;
    tilingDatafromBin->opBaseInfo.singleCoreLnBsSize = 7;
    tilingDatafromBin->mmTilingParams = {1, 7, 96,   32,   32,   7, 96, 32, 16, 96, 32, 1, 1, 1, 1, 1, 3072,
                                         0, 0, 3072, 6144, 3456, 1, 1,  1,  1,  1,  1,  0, 0, 2, 2, 1, 0,
                                         0, 0, 0,    0,    0,    0, 0,  0,  0,  0,  0,  0, 0, 0, 0, 0};
    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(100000UL);
    ICPU_RUN_KF(swin_transformer_ln_qkv_quant, blockDim, x, gamma, beta, weight, bias, quant_scale, quant_offset,
                dequant_scale, q, k, v, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(quant_scale);
    AscendC::GmFree(quant_offset);
    AscendC::GmFree(dequant_scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    free(path_);
}
