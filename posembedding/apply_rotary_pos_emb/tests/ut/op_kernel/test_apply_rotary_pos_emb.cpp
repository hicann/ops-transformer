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
#include "apply_rotary_pos_emb_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qout,
                                                           GM_ADDR kout, GM_ADDR workspace, GM_ADDR tiling);

class apply_rotary_pos_emb_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
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

TEST_F(apply_rotary_pos_emb_test, test_case_2)
{
    size_t inputqByteSize = 4 * 1024 * 16 * 64 * sizeof(int16_t);
    size_t inputkByteSize = 4 * 1024 * 16 * 64 * sizeof(int16_t);
    size_t outputByteSize = 4 * 1024 * 16 * 64 * sizeof(int16_t);
    size_t cosByteSize = 4 * 1024 * 1 * 64 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(ApplyRotaryPosEmbTilingData);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(inputqByteSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosByteSize);
    uint8_t* qout = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* kout = (uint8_t*)AscendC::GmAlloc(inputkByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // Check allocations
    if (!q || !k || !cos || !sin || !qout || !kout || !workspace || !tiling) {
        if (q)
            AscendC::GmFree(q);
        if (k)
            AscendC::GmFree(k);
        if (cos)
            AscendC::GmFree(cos);
        if (sin)
            AscendC::GmFree(sin);
        if (qout)
            AscendC::GmFree(qout);
        if (kout)
            AscendC::GmFree(kout);
        if (workspace)
            AscendC::GmFree(workspace);
        if (tiling)
            AscendC::GmFree(tiling);
        return;
    }

    uint32_t blockDim = 40;
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 4 1024 16 64 16 float16");
    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 40;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 9;
    tilingDatafromBin->preCBatchL = 4;
    tilingDatafromBin->lastCBatchL = 7;
    tilingDatafromBin->comBatchBB = 9;
    tilingDatafromBin->comBatchBBL = 9;
    tilingDatafromBin->comBatchBLL = 4;
    tilingDatafromBin->comBatchLLL = 7;
    tilingDatafromBin->qPart1Ub = 36864;
    tilingDatafromBin->q2q1Part1Ub = 36864;
    tilingDatafromBin->cosPart1Ub = 1152;
    tilingDatafromBin->sin1UbSize = 1152;
    tilingDatafromBin->preCLTimes = 11;
    tilingDatafromBin->lastCLTimes = 8;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 105472;
    tilingDatafromBin->kCoreOffset = 105472;
    tilingDatafromBin->cosCoreOffset = 6592;
    tilingDatafromBin->qcdNum = 1024;
    tilingDatafromBin->kcdNum = 1024;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 32;
    tilingDatafromBin->mulNum = 128;
    tilingDatafromBin->qcdHalfNum = 512;
    tilingDatafromBin->dstRepSBr = 4;
    tilingDatafromBin->blockLenQ = 64;
    tilingDatafromBin->srcStrideK = 64;
    tilingDatafromBin->blockLenq2q1 = 2;
    tilingDatafromBin->mask = 128;

    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(4);
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

TEST_F(apply_rotary_pos_emb_test, test_case_3)
{
    size_t inputqByteSize = 24 * 1 * 11 * 64 * sizeof(int16_t);
    size_t inputkByteSize = 24 * 1 * 1 * 64 * sizeof(int16_t);
    size_t outputByteSize = 24 * 1 * 11 * 64 * sizeof(int16_t);
    size_t cosByteSize = 24 * 1 * 1 * 64 * sizeof(int16_t);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 1 11 64 1 float16");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 24;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 0;
    tilingDatafromBin->preCBatchL = 0;
    tilingDatafromBin->lastCBatchL = 0;
    tilingDatafromBin->comBatchBB = 0;
    tilingDatafromBin->comBatchBBL = 0;
    tilingDatafromBin->comBatchBLL = 0;
    tilingDatafromBin->comBatchLLL = 0;
    tilingDatafromBin->qPart1Ub = 1536;
    tilingDatafromBin->q2q1Part1Ub = 1536;
    tilingDatafromBin->cosPart1Ub = 128;
    tilingDatafromBin->sin1UbSize = 128;
    tilingDatafromBin->preCLTimes = 0;
    tilingDatafromBin->lastCLTimes = 0;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 704;
    tilingDatafromBin->kCoreOffset = 64;
    tilingDatafromBin->cosCoreOffset = 64;
    tilingDatafromBin->qcdNum = 704;
    tilingDatafromBin->kcdNum = 64;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 12;
    tilingDatafromBin->mulNum = 768;
    tilingDatafromBin->qcdHalfNum = 1;
    tilingDatafromBin->dstRepSBr = 4;
    tilingDatafromBin->blockLenQ = 2;
    tilingDatafromBin->srcStrideK = 0;
    tilingDatafromBin->blockLenq2q1 = 0;
    tilingDatafromBin->mask = 64;
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

TEST_F(apply_rotary_pos_emb_test, test_case_4)
{
    size_t inputqByteSize = 24 * 1 * 11 * 64 * sizeof(float);
    size_t inputkByteSize = 24 * 1 * 1 * 64 * sizeof(float);
    size_t outputByteSize = 24 * 1 * 11 * 64 * sizeof(float);
    size_t cosByteSize = 24 * 1 * 1 * 64 * sizeof(float);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 1 11 64 1 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 24;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
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
    tilingDatafromBin->qCoreOffset = 704;
    tilingDatafromBin->kCoreOffset = 64;
    tilingDatafromBin->cosCoreOffset = 64;
    tilingDatafromBin->qcdNum = 704;
    tilingDatafromBin->kcdNum = 64;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 12;
    tilingDatafromBin->mulNum = 768;
    tilingDatafromBin->qcdHalfNum = 1;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 4;
    tilingDatafromBin->srcStrideK = 0;
    tilingDatafromBin->blockLenq2q1 = 0;
    tilingDatafromBin->mask = 64;
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

TEST_F(apply_rotary_pos_emb_test, test_case_5)
{
    size_t inputqByteSize = 24 * 1 * 11 * 64 * sizeof(int16_t);
    size_t inputkByteSize = 24 * 1 * 1 * 64 * sizeof(int16_t);
    size_t outputByteSize = 24 * 1 * 11 * 64 * sizeof(int16_t);
    size_t cosByteSize = 24 * 1 * 1 * 64 * sizeof(int16_t);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 1 11 64 1 bfloat16");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 24;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 0;
    tilingDatafromBin->preCBatchL = 0;
    tilingDatafromBin->lastCBatchL = 0;
    tilingDatafromBin->comBatchBB = 0;
    tilingDatafromBin->comBatchBBL = 0;
    tilingDatafromBin->comBatchBLL = 0;
    tilingDatafromBin->comBatchLLL = 0;
    tilingDatafromBin->qPart1Ub = 3072;
    tilingDatafromBin->q2q1Part1Ub = 3072;
    tilingDatafromBin->cosPart1Ub = 128;
    tilingDatafromBin->sin1UbSize = 256;
    tilingDatafromBin->preCLTimes = 0;
    tilingDatafromBin->lastCLTimes = 0;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 704;
    tilingDatafromBin->kCoreOffset = 64;
    tilingDatafromBin->cosCoreOffset = 64;
    tilingDatafromBin->qcdNum = 704;
    tilingDatafromBin->kcdNum = 64;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 12;
    tilingDatafromBin->mulNum = 768;
    tilingDatafromBin->qcdHalfNum = 1;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 4;
    tilingDatafromBin->srcStrideK = 0;
    tilingDatafromBin->blockLenq2q1 = 0;
    tilingDatafromBin->mask = 64;
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

TEST_F(apply_rotary_pos_emb_test, test_case_6)
{
    size_t inputqByteSize = 4096 * 60 * 17 * 64 * sizeof(int16_t);
    size_t inputkByteSize = 4096 * 60 * 11 * 64 * sizeof(int16_t);
    size_t outputByteSize = 4096 * 60 * 17 * 64 * sizeof(int16_t);
    size_t cosByteSize = 4096 * 60 * 1 * 64 * sizeof(int16_t);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 4096 60 17 64 11 float16");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 40;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 11;
    tilingDatafromBin->preCBatchL = 6;
    tilingDatafromBin->lastCBatchL = 6;
    tilingDatafromBin->comBatchBB = 10;
    tilingDatafromBin->comBatchBBL = 1;
    tilingDatafromBin->comBatchBLL = 6;
    tilingDatafromBin->comBatchLLL = 6;
    tilingDatafromBin->qPart1Ub = 39424;
    tilingDatafromBin->q2q1Part1Ub = 35840;
    tilingDatafromBin->cosPart1Ub = 1408;
    tilingDatafromBin->sin1UbSize = 1280;
    tilingDatafromBin->preCLTimes = 558;
    tilingDatafromBin->lastCLTimes = 558;
    tilingDatafromBin->preCBBTimes = 1;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 6684672;
    tilingDatafromBin->kCoreOffset = 4325376;
    tilingDatafromBin->cosCoreOffset = 393216;
    tilingDatafromBin->qcdNum = 1088;
    tilingDatafromBin->kcdNum = 704;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 28;
    tilingDatafromBin->mulNum = 112;
    tilingDatafromBin->qcdHalfNum = 544;
    tilingDatafromBin->dstRepSBr = 4;
    tilingDatafromBin->blockLenQ = 68;
    tilingDatafromBin->srcStrideK = 44;
    tilingDatafromBin->blockLenq2q1 = 2;
    tilingDatafromBin->mask = 128;
    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(3);
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

TEST_F(apply_rotary_pos_emb_test, test_case_7)
{
    size_t inputqByteSize = 24 * 60 * 11 * 64 * sizeof(int16_t);
    size_t inputkByteSize = 4096 * 60 * 1 * 64 * sizeof(int16_t);
    size_t outputByteSize = 24 * 60 * 11 * 64 * sizeof(int16_t);
    size_t cosByteSize = 24 * 60 * 1 * 64 * sizeof(int16_t);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 60 11 64 1 float16");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 40;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 24;
    tilingDatafromBin->preCBatchL = 12;
    tilingDatafromBin->lastCBatchL = 12;
    tilingDatafromBin->comBatchBB = 24;
    tilingDatafromBin->comBatchBBL = 24;
    tilingDatafromBin->comBatchBLL = 12;
    tilingDatafromBin->comBatchLLL = 12;
    tilingDatafromBin->qPart1Ub = 36864;
    tilingDatafromBin->q2q1Part1Ub = 36864;
    tilingDatafromBin->cosPart1Ub = 3072;
    tilingDatafromBin->sin1UbSize = 3072;
    tilingDatafromBin->preCLTimes = 1;
    tilingDatafromBin->lastCLTimes = 1;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 25344;
    tilingDatafromBin->kCoreOffset = 2304;
    tilingDatafromBin->cosCoreOffset = 2304;
    tilingDatafromBin->qcdNum = 704;
    tilingDatafromBin->kcdNum = 64;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 12;
    tilingDatafromBin->mulNum = 48;
    tilingDatafromBin->qcdHalfNum = 352;
    tilingDatafromBin->dstRepSBr = 4;
    tilingDatafromBin->blockLenQ = 44;
    tilingDatafromBin->srcStrideK = 4;
    tilingDatafromBin->blockLenq2q1 = 2;
    tilingDatafromBin->mask = 128;
    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(4);
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

TEST_F(apply_rotary_pos_emb_test, test_case_8)
{
    size_t inputqByteSize = 24 * 13 * 128 * sizeof(float);
    size_t inputkByteSize = 24 * 13 * 128 * sizeof(float);
    size_t outputByteSize = 24 * 1 * 64 * sizeof(float);
    size_t cosByteSize = 24 * 1 * 64 * sizeof(float);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 24 13 128 13 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 24;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 0;
    tilingDatafromBin->preCBatchL = 0;
    tilingDatafromBin->lastCBatchL = 0;
    tilingDatafromBin->comBatchBB = 0;
    tilingDatafromBin->comBatchBBL = 0;
    tilingDatafromBin->comBatchBLL = 0;
    tilingDatafromBin->comBatchLLL = 0;
    tilingDatafromBin->qPart1Ub = 6656;
    tilingDatafromBin->q2q1Part1Ub = 6656;
    tilingDatafromBin->cosPart1Ub = 256;
    tilingDatafromBin->sin1UbSize = 256;
    tilingDatafromBin->preCLTimes = 0;
    tilingDatafromBin->lastCLTimes = 0;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 1664;
    tilingDatafromBin->kCoreOffset = 1664;
    tilingDatafromBin->cosCoreOffset = 64;
    tilingDatafromBin->qcdNum = 832;
    tilingDatafromBin->kcdNum = 832;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 26;
    tilingDatafromBin->mulNum = 1664;
    tilingDatafromBin->qcdHalfNum = 1;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 4;
    tilingDatafromBin->srcStrideK = 0;
    tilingDatafromBin->blockLenq2q1 = 0;
    tilingDatafromBin->mask = 64;
    tilingDatafromBin->qcNum = 13;
    tilingDatafromBin->kcNum = 13;
    tilingDatafromBin->qDim3 = 128;
    tilingDatafromBin->kDim3 = 128;
    tilingDatafromBin->blockMoveQ = 8;
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

TEST_F(apply_rotary_pos_emb_test, test_case_9)
{
    size_t inputqByteSize = 90 * 47 * 128 * sizeof(float);
    size_t inputkByteSize = 90 * 1 * 128 * sizeof(float);
    size_t outputByteSize = 90 * 1 * 64 * sizeof(float);
    size_t cosByteSize = 90 * 1 * 64 * sizeof(float);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 90 47 128 1 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 45;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 2;
    tilingDatafromBin->preCBatchL = 2;
    tilingDatafromBin->lastCBatchL = 2;
    tilingDatafromBin->comBatchBB = 2;
    tilingDatafromBin->comBatchBBL = 2;
    tilingDatafromBin->comBatchBLL = 2;
    tilingDatafromBin->comBatchLLL = 2;
    tilingDatafromBin->qPart1Ub = 24576;
    tilingDatafromBin->q2q1Part1Ub = 24576;
    tilingDatafromBin->cosPart1Ub = 512;
    tilingDatafromBin->sin1UbSize = 512;
    tilingDatafromBin->preCLTimes = 0;
    tilingDatafromBin->lastCLTimes = 0;
    tilingDatafromBin->preCBBTimes = 0;
    tilingDatafromBin->preCBLTimes = 0;
    tilingDatafromBin->preCLLTimes = 0;
    tilingDatafromBin->qCoreOffset = 12032;
    tilingDatafromBin->kCoreOffset = 256;
    tilingDatafromBin->cosCoreOffset = 128;
    tilingDatafromBin->qcNum = 47;
    tilingDatafromBin->kcNum = 1;
    tilingDatafromBin->qcdNum = 3008;
    tilingDatafromBin->kcdNum = 64;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 48;
    tilingDatafromBin->mulNum = 384;
    tilingDatafromBin->qcdHalfNum = 1504;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 376;
    tilingDatafromBin->srcStrideK = 8;
    tilingDatafromBin->blockLenq2q1 = 4;
    tilingDatafromBin->mask = 64;
    tilingDatafromBin->qDim3 = 128;
    tilingDatafromBin->kDim3 = 128;
    tilingDatafromBin->blockMoveQ = 8;
    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(4);
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

TEST_F(apply_rotary_pos_emb_test, test_case_10)
{
    size_t inputqByteSize = 9000 * 46 * 64 * sizeof(float);
    size_t inputkByteSize = 9000 * 48 * 64 * sizeof(float);
    size_t outputByteSize = 9000 * 1 * 64 * sizeof(float);
    size_t cosByteSize = 9000 * 1 * 64 * sizeof(float);
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
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb/tests/ut/op_kernel/apply_rotary_pos_emb_data "
           "./");
    system("chmod -R 755 ./apply_rotary_pos_emb_data/");
    system("cd ./apply_rotary_pos_emb_data/ && rm -rf ./*bin");
    system("cd ./apply_rotary_pos_emb_data/ && python3 gen_data.py 9000 46 64 48 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyRotaryPosEmbTilingData* tilingDatafromBin = reinterpret_cast<ApplyRotaryPosEmbTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 48;
    tilingDatafromBin->lastDim = 64;
    tilingDatafromBin->halfNum = 32;
    tilingDatafromBin->preCBatchB = 2;
    tilingDatafromBin->preCBatchL = 2;
    tilingDatafromBin->lastCBatchL = 2;
    tilingDatafromBin->comBatchBB = 1;
    tilingDatafromBin->comBatchBBL = 1;
    tilingDatafromBin->comBatchBLL = 1;
    tilingDatafromBin->comBatchLLL = 1;
    tilingDatafromBin->qPart1Ub = 48128;
    tilingDatafromBin->q2q1Part1Ub = 24064;
    tilingDatafromBin->cosPart1Ub = 512;
    tilingDatafromBin->sin1UbSize = 256;
    tilingDatafromBin->preCLTimes = 93;
    tilingDatafromBin->lastCLTimes = 81;
    tilingDatafromBin->preCBBTimes = 1;
    tilingDatafromBin->preCBLTimes = 1;
    tilingDatafromBin->preCLLTimes = 1;
    tilingDatafromBin->qCoreOffset = 553472;
    tilingDatafromBin->kCoreOffset = 577536;
    tilingDatafromBin->cosCoreOffset = 12032;
    tilingDatafromBin->qcNum = 46;
    tilingDatafromBin->kcNum = 48;
    tilingDatafromBin->qcdNum = 2944;
    tilingDatafromBin->kcdNum = 3072;
    tilingDatafromBin->coscdNum = 64;
    tilingDatafromBin->qkcNum = 94;
    tilingDatafromBin->mulNum = 752;
    tilingDatafromBin->qcdHalfNum = 1472;
    tilingDatafromBin->dstRepSBr = 8;
    tilingDatafromBin->blockLenQ = 368;
    tilingDatafromBin->srcStrideK = 384;
    tilingDatafromBin->blockLenq2q1 = 4;
    tilingDatafromBin->mask = 64;
    tilingDatafromBin->qDim3 = 64;
    tilingDatafromBin->kDim3 = 64;
    tilingDatafromBin->blockMoveQ = 8;
    ReadFile(path + "/apply_rotary_pos_emb_data/q.bin", inputqByteSize, q, inputqByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/k.bin", inputkByteSize, k, inputkByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_data/sin.bin", cosByteSize, sin, cosByteSize);
    ICPU_SET_TILING_KEY(3);
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
