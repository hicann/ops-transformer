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
 * \file test_rope_quant_kvcache.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "rope_quant_kvcache_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rope_quant_kvcache(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale,
                                                         GM_ADDR quant_offset, GM_ADDR k_cache, GM_ADDR v_cache,
                                                         GM_ADDR indice, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
                                                         GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR workspace,
                                                         GM_ADDR tiling);

class rope_quant_kvcache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rope_quant_kvcache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rope_quant_kvcache_test TearDown\n" << endl;
    }
};

TEST_F(rope_quant_kvcache_test, test_case_half)
{
    size_t qkv_size = 4 * 1 * 1280 * 2;
    size_t cos_size = 4 * 1 * 1 * 128 * 2;
    size_t sin_size = 4 * 1 * 1 * 128 * 2;
    size_t scale_size = 128 * 4;
    size_t offset_size = 128 * 4;
    size_t k_cache_size = 4 * 2048 * 1 * 128;
    size_t v_cache_size = 4 * 2048 * 1 * 128;
    size_t indice_size = 4 * 1 * 4;
    size_t q_out_size = 4 * 1 * 8 * 128 * 2;
    size_t k_out_size = 4 * 1 * 1 * 128 * 2;
    size_t v_out_size = 4 * 1 * 1 * 128 * 2;
    size_t k_cache_out_size = 4 * 2048 * 1 * 128;
    size_t v_cache_out_size = 4 * 2048 * 1 * 128;

    size_t tiling_data_size = sizeof(RopeQuantKvcacheTilingData);
    uint32_t blockDim = 4;

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkv_size);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cos_size);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sin_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offset_size);
    uint8_t* k_cache = (uint8_t*)AscendC::GmAlloc(k_cache_size);
    uint8_t* v_cache = (uint8_t*)AscendC::GmAlloc(v_cache_size);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indice_size);

    uint8_t* q_out = (uint8_t*)AscendC::GmAlloc(q_out_size);
    uint8_t* k_out = (uint8_t*)AscendC::GmAlloc(k_out_size);
    uint8_t* v_out = (uint8_t*)AscendC::GmAlloc(v_out_size);
    uint8_t* k_cache_out = (uint8_t*)AscendC::GmAlloc(k_cache_out_size);
    uint8_t* v_cache_out = (uint8_t*)AscendC::GmAlloc(v_cache_out_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // system(
    // "cp -r "
    // "../../../../../../../ops/built-in/tests/ut/fast_op_test/stft/stft_data
    // "
    // "./");
    // system("chmod -R 755 ./stft_data/");
    // system("cd ./stft_data/ && rm -rf ./*bin");
    // system("cd ./stft_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./stft_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    RopeQuantKvcacheTilingData* tilingDatafromBin = reinterpret_cast<RopeQuantKvcacheTilingData*>(tiling);
    tilingDatafromBin->qHeadNum = 8;
    tilingDatafromBin->kvHeadNum = 1;
    tilingDatafromBin->hiddenSize = 128;
    tilingDatafromBin->cacheSeqlen = 2048;
    tilingDatafromBin->qHiddenSize = 8 * 128;
    tilingDatafromBin->kHiddenSize = 128;
    tilingDatafromBin->vHiddenSize = 128;

    ICPU_RUN_KF(rope_quant_kvcache, blockDim, qkv, cos, sin, scale, offset, k_cache, v_cache, indice, q_out, k_out,
                v_out, k_cache_out, v_cache_out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(qkv);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(k_cache);
    AscendC::GmFree(v_cache);
    AscendC::GmFree(indice);

    AscendC::GmFree(q_out);
    AscendC::GmFree(k_out);
    AscendC::GmFree(v_out);
    AscendC::GmFree(k_cache_out);
    AscendC::GmFree(v_cache_out);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
