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
#include "apply_rotary_pos_emb_tiling_regbase.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qout,
                                                           GM_ADDR kout, GM_ADDR workspace, GM_ADDR tiling);

class apply_rotary_pos_emb_regbase_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "apply_rotary_pos_emb_regbase_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "apply_rotary_pos_emb_regbase_test TearDown\n" << endl;
    }
};

// TEST_F(apply_rotary_pos_emb_regbase_test, test_case_b1sd)
// {
//     size_t inputqByteSize = 24 * 11 * 1 * 128 * sizeof(int16_t);
//     size_t inputkByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 11 * 1 * 128 * sizeof(int16_t);
//     size_t cosByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(ApplyRotaryPosEmbRegbaseTilingData);
//     uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
//     uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     ICPU_SET_TILING_KEY(TILING_KEY_ABA);
//     ICPU_RUN_KF(apply_rotary_pos_emb, blockDim, q, k, cos, sin, qout, kout, workspace, tiling);

//     AscendC::GmFree(q);
//     AscendC::GmFree(k);
//     AscendC::GmFree(cos);
//     AscendC::GmFree(sin);
//     AscendC::GmFree(qout);
//     AscendC::GmFree(kout);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(apply_rotary_pos_emb_regbase_test, test_case_11sd)
// {
//     size_t inputqByteSize = 24 * 11 * 1 * 128 * sizeof(int16_t);
//     size_t inputkByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 11 * 1 * 128 * sizeof(int16_t);
//     size_t cosByteSize = 1 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(ApplyRotaryPosEmbRegbaseTilingData);
//     uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
//     uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     ICPU_SET_TILING_KEY(TILING_KEY_BA);
//     ICPU_RUN_KF(apply_rotary_pos_emb, blockDim, q, k, cos, sin, qout, kout, workspace, tiling);

//     AscendC::GmFree(q);
//     AscendC::GmFree(k);
//     AscendC::GmFree(cos);
//     AscendC::GmFree(sin);
//     AscendC::GmFree(qout);
//     AscendC::GmFree(kout);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(apply_rotary_pos_emb_regbase_test, test_case_1s1d)
// {
//     size_t inputqByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
//     size_t inputkByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
//     size_t cosByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(ApplyRotaryPosEmbRegbaseTilingData);
//     uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
//     uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     ICPU_SET_TILING_KEY(TILING_KEY_BAB);
//     ICPU_RUN_KF(apply_rotary_pos_emb, blockDim, q, k, cos, sin, qout, kout, workspace, tiling);

//     AscendC::GmFree(q);
//     AscendC::GmFree(k);
//     AscendC::GmFree(cos);
//     AscendC::GmFree(sin);
//     AscendC::GmFree(qout);
//     AscendC::GmFree(kout);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(apply_rotary_pos_emb_regbase_test, test_case_aba)
// {
//     size_t inputqByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
//     size_t inputkByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
//     size_t cosByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(ApplyRotaryPosEmbRegbaseABTilingData);
//     uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
//     uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
//     uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     ICPU_SET_TILING_KEY(TILING_KEY_AB);
//     ICPU_RUN_KF(apply_rotary_pos_emb, blockDim, q, k, cos, sin, qout, kout, workspace, tiling);

//     AscendC::GmFree(q);
//     AscendC::GmFree(k);
//     AscendC::GmFree(cos);
//     AscendC::GmFree(sin);
//     AscendC::GmFree(qout);
//     AscendC::GmFree(kout);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }
