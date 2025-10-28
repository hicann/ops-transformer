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
#include "apply_rotary_pos_emb_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qout, GM_ADDR kout, GM_ADDR workspace, GM_ADDR tiling);

class apply_rotary_pos_emb_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "apply_rotary_pos_emb_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "apply_rotary_pos_emb_test TearDown\n" << endl;
    }
};

TEST_F(apply_rotary_pos_emb_test, test_case_1)
{
    size_t inputqByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
    size_t inputkByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
    size_t outputByteSize = 24 * 1 * 11 * 128 * sizeof(int16_t);
    size_t cosByteSize = 24 * 1 * 1 * 128 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(ApplyRotaryPosEmbTilingData);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
    uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;
    system(
        "cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
        "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 1 11 128 1 float16");

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 24;
    tilingDatafromBin->lastDim = 128;
    tilingDatafromBin->halfNum = 64;
    tilingDatafromBin->preCBatchB = 0;
    tilingDatafromBin->preCBatchL = 0;
    tilingDatafromBin->lastCBatchL = 0;
    tilingDatafromBin->comBatchBB = 0;
    tilingDatafromBin->comBatchBBL = 0;
    tilingDatafromBin->comBatchBLL = 0;
    tilingDatafromBin->comBatchLLL = 0;
    tilingDatafromBin->qPart1Ub = 3072;
    tilingDatafromBin->q2q1Part1Ub = 3072;
    tilingDatafromBin->cosPart1Ub = 256;
    tilingDatafromBin->sin1UbSize = 256;
    tilingDatafromBin->preCLTimes = 0;
    tilingDatafromBin->lastCLTimes = 0;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 1408;
    tilingDatafromBin->kCoreOffset = 128;
    tilingDatafromBin->cosCoreOffset = 128;
    tilingDatafromBin->qcNum = 11;
    tilingDatafromBin->kcNum = 1;
    tilingDatafromBin->coscNum = 1;
    tilingDatafromBin->qcdNum = 1408;
    tilingDatafromBin->kcdNum = 128;
    tilingDatafromBin->coscdNum = 128;
    tilingDatafromBin->qkcNum = 12;
    tilingDatafromBin->mulNum = 1536;
    tilingDatafromBin->qcdHalfNum = 1;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 4;
    tilingDatafromBin->srcStrideK = 0;
    tilingDatafromBin->blockLenq2q1 = 0;
    tilingDatafromBin->mask = 128;
    tilingDatafromBin->tilingKey = 1;
    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(apply_rotary_pos_emb, blockDim, q, k, cos, sin, qout, kout, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(qout);
    AscendC::GmFree(kout);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
