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
#include "moe_gating_top_k_softmax_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax(
    GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR sourceRowsOut, GM_ADDR workspace,
    GM_ADDR tiling);

class moe_gating_top_k_softmax_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_gating_top_k_softmax_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_gating_top_k_softmax_test TearDown\n" << endl;
    }
};

TEST_F(moe_gating_top_k_softmax_test, test_case_float32)
{
    size_t inputByteSize = 2 * 48 * 32 * sizeof(float);
    size_t finishedByteSize = 2 * 48 * 32 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 32 * sizeof(float);
    size_t outputByteSize1 = 2 * 48 * 16 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxEKFullLoadTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxEKFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxEKFullLoadTilingData*>(tiling);

    SoftMaxTiling SoftMaxTilingData;
    TopkTiling TopkTilingData;

    tilingDatafromBin->tilingKey = 1;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 32;
    tilingDatafromBin->colAlign = 32;
    tilingDatafromBin->k = 16;
    tilingDatafromBin->kAlignB16 = 32;
    tilingDatafromBin->kAlignB32 = 64;
    tilingDatafromBin->blockNum = 24;
    tilingDatafromBin->blockFormer = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 4;
    tilingDatafromBin->ubTailOfFormerBlock = 4;
    tilingDatafromBin->ubTailOfTailBlock = 4;

    SoftMaxTilingData.srcM = 4;
    SoftMaxTilingData.srcK = 32;
    SoftMaxTilingData.srcSize = 128;
    SoftMaxTilingData.outMaxM = 4;
    SoftMaxTilingData.outMaxK = 8;
    SoftMaxTilingData.outMaxSize = 32;
    SoftMaxTilingData.splitM = 4;
    SoftMaxTilingData.splitK = 32;
    SoftMaxTilingData.splitSize = 128;
    SoftMaxTilingData.reduceM = 4;
    SoftMaxTilingData.reduceK = 8;
    SoftMaxTilingData.reduceSize = 32;
    SoftMaxTilingData.rangeM = 1;
    SoftMaxTilingData.tailM = 0;
    SoftMaxTilingData.tailSplitSize = 0;
    SoftMaxTilingData.tailReduceSize = 0;

    TopkTilingData.tmpLocalSize = 128;
    TopkTilingData.allDataSize = 128;
    TopkTilingData.innerDataSize = 0;
    TopkTilingData.sortRepeat = 1;
    TopkTilingData.mrgSortRepeat = 1;
    TopkTilingData.kAlignFourBytes = 16;
    TopkTilingData.kAlignTwoBytes = 16;
    TopkTilingData.maskOffset = 64;
    TopkTilingData.maskVreducev2FourBytes = 32;
    TopkTilingData.maskVreducev2TwoBytes = 64;
    TopkTilingData.mrgSortSrc1offset = 2;
    TopkTilingData.mrgSortSrc2offset = 4;
    TopkTilingData.mrgSortSrc3offset = 6;
    TopkTilingData.mrgSortTwoQueueSrc1Offset = 2;
    TopkTilingData.mrgFourQueueTailPara1 = 64;
    TopkTilingData.mrgFourQueueTailPara2 = 1;
    TopkTilingData.srcIndexOffset = 0;
    TopkTilingData.topkMrgSrc1MaskSizeOffset = 0;
    TopkTilingData.topkNSmallSrcIndexOffset = 0;
    TopkTilingData.vreduceValMask0 = 0;
    TopkTilingData.vreduceValMask1 = 0;
    TopkTilingData.vreduceIdxMask0 = 0;
    TopkTilingData.vreduceIdxMask1 = 0;
    TopkTilingData.vreducehalfValMask0 = 0;
    TopkTilingData.vreducehalfValMask1 = 0;
    TopkTilingData.vreducehalfValMask2 = 0;
    TopkTilingData.vreducehalfValMask3 = 0;
    TopkTilingData.vreducehalfValMask4 = 0;
    TopkTilingData.vreducehalfValMask5 = 0;
    TopkTilingData.vreducehalfValMask6 = 0;
    TopkTilingData.vreducehalfValMask7 = 0;
    TopkTilingData.copyUbToUbBlockCount = 1;
    tilingDatafromBin->formerSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->formerBlockTailSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->tailBlockTailSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->formerTopkTilingData = TopkTilingData;
    tilingDatafromBin->formerBlockTailTopkTilingData = TopkTilingData;
    tilingDatafromBin->tailBlockTailTopkTilingData = TopkTilingData;
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, finished, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_bf16)
{
    size_t inputByteSize = 2 * 48 * 32 * sizeof(float);
    size_t finishedByteSize = 2 * 48 * 32 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 32 * sizeof(float);
    size_t outputByteSize1 = 2 * 48 * 16 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxEKFullLoadTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxEKFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxEKFullLoadTilingData*>(tiling);

    SoftMaxTiling SoftMaxTilingData;
    TopkTiling TopkTilingData;

    tilingDatafromBin->tilingKey = 5;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 32;
    tilingDatafromBin->colAlign = 32;
    tilingDatafromBin->k = 16;
    tilingDatafromBin->kAlignB16 = 32;
    tilingDatafromBin->kAlignB32 = 64;
    tilingDatafromBin->blockNum = 24;
    tilingDatafromBin->blockFormer = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 4;
    tilingDatafromBin->ubTailOfFormerBlock = 4;
    tilingDatafromBin->ubTailOfTailBlock = 4;

    SoftMaxTilingData.srcM = 4;
    SoftMaxTilingData.srcK = 32;
    SoftMaxTilingData.srcSize = 128;
    SoftMaxTilingData.outMaxM = 4;
    SoftMaxTilingData.outMaxK = 8;
    SoftMaxTilingData.outMaxSize = 32;
    SoftMaxTilingData.splitM = 4;
    SoftMaxTilingData.splitK = 32;
    SoftMaxTilingData.splitSize = 128;
    SoftMaxTilingData.reduceM = 4;
    SoftMaxTilingData.reduceK = 8;
    SoftMaxTilingData.reduceSize = 32;
    SoftMaxTilingData.rangeM = 1;
    SoftMaxTilingData.tailM = 0;
    SoftMaxTilingData.tailSplitSize = 0;
    SoftMaxTilingData.tailReduceSize = 0;

    TopkTilingData.tmpLocalSize = 128;
    TopkTilingData.allDataSize = 128;
    TopkTilingData.innerDataSize = 0;
    TopkTilingData.sortRepeat = 1;
    TopkTilingData.mrgSortRepeat = 1;
    TopkTilingData.kAlignFourBytes = 16;
    TopkTilingData.kAlignTwoBytes = 16;
    TopkTilingData.maskOffset = 64;
    TopkTilingData.maskVreducev2FourBytes = 32;
    TopkTilingData.maskVreducev2TwoBytes = 64;
    TopkTilingData.mrgSortSrc1offset = 2;
    TopkTilingData.mrgSortSrc2offset = 4;
    TopkTilingData.mrgSortSrc3offset = 6;
    TopkTilingData.mrgSortTwoQueueSrc1Offset = 2;
    TopkTilingData.mrgFourQueueTailPara1 = 64;
    TopkTilingData.mrgFourQueueTailPara2 = 1;
    TopkTilingData.srcIndexOffset = 0;
    TopkTilingData.topkMrgSrc1MaskSizeOffset = 0;
    TopkTilingData.topkNSmallSrcIndexOffset = 0;
    TopkTilingData.vreduceValMask0 = 0;
    TopkTilingData.vreduceValMask1 = 0;
    TopkTilingData.vreduceIdxMask0 = 0;
    TopkTilingData.vreduceIdxMask1 = 0;
    TopkTilingData.vreducehalfValMask0 = 0;
    TopkTilingData.vreducehalfValMask1 = 0;
    TopkTilingData.vreducehalfValMask2 = 0;
    TopkTilingData.vreducehalfValMask3 = 0;
    TopkTilingData.vreducehalfValMask4 = 0;
    TopkTilingData.vreducehalfValMask5 = 0;
    TopkTilingData.vreducehalfValMask6 = 0;
    TopkTilingData.vreducehalfValMask7 = 0;
    TopkTilingData.copyUbToUbBlockCount = 1;
    tilingDatafromBin->formerSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->formerBlockTailSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->tailBlockTailSoftmaxTilingData = SoftMaxTilingData;
    tilingDatafromBin->formerTopkTilingData = TopkTilingData;
    tilingDatafromBin->formerBlockTailTopkTilingData = TopkTilingData;
    tilingDatafromBin->tailBlockTailTopkTilingData = TopkTilingData;

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, finished, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_float32_small_col_num)
{
    size_t inputByteSize = 2 * 48 * 8 * sizeof(float);
    size_t finishedByteSize = 2 * 48 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 8 * sizeof(float);
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 9;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 8;
    tilingDatafromBin->colAlign = 32;
    tilingDatafromBin->colBytesAlign = 8;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 96;

    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_float32_middle_col_num)
{
    size_t inputByteSize = 2 * 48 * 64 * sizeof(float);
    size_t finishedByteSize = 2 * 48 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 64 * sizeof(float);
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 12;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 64;
    tilingDatafromBin->colAlign = 64;
    tilingDatafromBin->colBytesAlign = 64;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 192;

    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_float32_big_col_num)
{
    size_t inputByteSize = 2 * 48 * 256 * sizeof(float);
    size_t finishedByteSize = 2 * 48 * 256 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 256 * sizeof(float);
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 15;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 256;
    tilingDatafromBin->colAlign = 256;
    tilingDatafromBin->colBytesAlign = 256;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 768;

    ICPU_SET_TILING_KEY(15);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_bfloat16_big_col_num)
{
    size_t inputByteSize = 2 * 48 * 256 * sizeof(float) / 2;
    size_t finishedByteSize = 2 * 48 * 256 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 256 * sizeof(float) / 2;
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 17;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 256;
    tilingDatafromBin->colAlign = 256;
    tilingDatafromBin->colBytesAlign = 256;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 768;

    ICPU_SET_TILING_KEY(17);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_bfloat16_middle_col_num)
{
    size_t inputByteSize = 2 * 48 * 64 * sizeof(float) / 2;
    size_t finishedByteSize = 2 * 48 * 64 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 64 * sizeof(float) / 2;
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 14;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 64;
    tilingDatafromBin->colAlign = 64;
    tilingDatafromBin->colBytesAlign = 64;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 192;

    ICPU_SET_TILING_KEY(14);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_gating_top_k_softmax_test, test_case_perf_bfloat16_small_col_num)
{
    size_t inputByteSize = 2 * 48 * 8 * sizeof(float) / 2;
    size_t finishedByteSize = 2 * 48 * 8 * sizeof(bool);
    size_t outputByteSize = 2 * 48 * 8 * sizeof(float) / 2;
    size_t outputByteSize1 = 2 * 48 * 4 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(MoeGatingTopKSoftmaxPerfTilingData);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* finished = (uint8_t*)AscendC::GmAlloc(finishedByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* row_idx = (uint8_t*)AscendC::GmAlloc(outputByteSize1);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeGatingTopKSoftmaxPerfTilingData* tilingDatafromBin =
        reinterpret_cast<MoeGatingTopKSoftmaxPerfTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 11;
    tilingDatafromBin->row = 96;
    tilingDatafromBin->col = 8;
    tilingDatafromBin->colAlign = 32;
    tilingDatafromBin->colBytesAlign = 8;
    tilingDatafromBin->k = 4;
    tilingDatafromBin->kAlign = 8;
    tilingDatafromBin->blockNum = 32;
    tilingDatafromBin->blockFormer = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubFormer = 3;
    tilingDatafromBin->ubTailOfFormerBlock = 3;
    tilingDatafromBin->ubTailOfTailBlock = 3;
    tilingDatafromBin->topKValuesMask = 85;
    tilingDatafromBin->topKIndicesMask = 170;
    tilingDatafromBin->bufferElemSize = 96;

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        moe_gating_top_k_softmax, blockDim, x, nullptr, y, expert_idx, row_idx, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(finished);
    AscendC::GmFree(y);
    AscendC::GmFree(expert_idx);
    AscendC::GmFree(row_idx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
