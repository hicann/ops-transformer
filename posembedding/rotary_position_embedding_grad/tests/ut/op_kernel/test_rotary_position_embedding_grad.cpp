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
#include "test_rotary_position_embedding_grad.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rotary_position_embedding_grad(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, GM_ADDR tiling);

class rotary_position_embedding_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding_grad_test TearDown\n" << endl;
    }
};

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp16_pad_needbackward)
{
    uint32_t B = 40, S = 1, N = 16, D = 8;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputCosByteSize = B * 1 * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradUTTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    system(
        "cp -r ../../../../../posembedding/rotary_position_embedding_grad/tests/ut/op_kernel/rotary_position_embedding_grad_data "
        "./");
    system("chmod -R 755 ./rotary_position_embedding_grad_data/");
    system("cd ./rotary_position_embedding_grad_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding_grad_data/ && python3 gen_data.py 40 1 16 8 float16");
    system("cd ./rotary_position_embedding_grad_data/ && python3 gen_tiling.py case0");
    char* path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbeddingGradUTTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradUTTilingData*>(tiling);
    ReadFile(path + "/rotary_position_embedding_grad_data/x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_grad_data/dy.bin", inputXByteSize, grad, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_grad_data/cos.bin", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_grad_data/sin.bin", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_grad_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(120);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
