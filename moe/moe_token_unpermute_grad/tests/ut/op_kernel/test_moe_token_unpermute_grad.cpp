/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_grad.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_moe_token_unpermute_grad.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_unpermute_grad(
    GM_ADDR permuted_tokens, GM_ADDR unpermuted_output_d, GM_ADDR sorted_indices, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace, GM_ADDR tiling);
class moe_token_unpermute_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_token_unpermute_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_token_unpermute_grad_test TearDown\n" << endl;
    }
};

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_bf16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 bfloat16_t bfloat16_t True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(bfloat16_t);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(bfloat16_t);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t probsGradByteSize = 10 * 3 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_bf16_fp32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 bfloat16_t float32 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(bfloat16_t);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(float);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t probsGradByteSize = 10 * 3 * sizeof(float);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_fp16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 float16 float16 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(half);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(half);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(half);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(half);
    size_t probsGradByteSize = 10 * 3 * sizeof(half);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_fp16_bf16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 float16 bfloat16_t True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(half);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(half);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(bfloat16_t);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(half);
    size_t probsGradByteSize = 10 * 3 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_fp32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 float32 float32 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(float);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(float);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(float);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(float);
    size_t probsGradByteSize = 10 * 3 * sizeof(float);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_fp32_bf16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 64 float32 bfloat16_t True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case0");

    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(float);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(float);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(bfloat16_t);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(float);
    size_t probsGradByteSize = 10 * 3 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_none_bf16)
{
    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(bfloat16_t);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(bfloat16_t);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(bfloat16_t);
    size_t probsGradByteSize = 10 * 3 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);
    tilingDatafromBin->tokensNum = 10;
    tilingDatafromBin->topK = 1;
    tilingDatafromBin->hiddenSize = 64;
    tilingDatafromBin->numOutTokens = 10;
    tilingDatafromBin->formerCoreNum = 10;
    tilingDatafromBin->tailCoreNum = 38;
    tilingDatafromBin->tokenNumEachCore = 1;
    tilingDatafromBin->tokenNumTailCore = 0;
    tilingDatafromBin->rowIdMapEachCore = 1;
    tilingDatafromBin->rowIdMapTailCore = 0;
    tilingDatafromBin->hiddenSizeAlign = 64;
    tilingDatafromBin->hiddenSizeLoopTimes = 1;
    tilingDatafromBin->hiddenSizeTail = 64;
    tilingDatafromBin->inputReserveNum = 1;
    tilingDatafromBin->indicesReserveNum = 1;
    tilingDatafromBin->indicesReserveNumAlign = 16;
    tilingDatafromBin->totalUbSize = 196352;
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_none_fp16)
{
    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(half);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(half);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(half);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(half);
    size_t probsGradByteSize = 10 * 3 * sizeof(half);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);
    tilingDatafromBin->tokensNum = 10;
    tilingDatafromBin->topK = 1;
    tilingDatafromBin->hiddenSize = 64;
    tilingDatafromBin->numOutTokens = 10;
    tilingDatafromBin->formerCoreNum = 10;
    tilingDatafromBin->tailCoreNum = 38;
    tilingDatafromBin->tokenNumEachCore = 1;
    tilingDatafromBin->tokenNumTailCore = 0;
    tilingDatafromBin->rowIdMapEachCore = 1;
    tilingDatafromBin->rowIdMapTailCore = 0;
    tilingDatafromBin->hiddenSizeAlign = 64;
    tilingDatafromBin->hiddenSizeLoopTimes = 1;
    tilingDatafromBin->hiddenSizeTail = 64;
    tilingDatafromBin->inputReserveNum = 1;
    tilingDatafromBin->indicesReserveNum = 1;
    tilingDatafromBin->indicesReserveNumAlign = 16;
    tilingDatafromBin->totalUbSize = 196352;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_none_fp32)
{
    size_t permutedTokensByteSize = 10 * 3 * 64 * sizeof(float);
    size_t unpermutedOutputDByteSize = 10 * 64 * sizeof(float);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(float);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 64 * sizeof(float);
    size_t probsGradByteSize = 10 * 3 * sizeof(float);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize + 32);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize + 32);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);
    tilingDatafromBin->tokensNum = 10;
    tilingDatafromBin->topK = 1;
    tilingDatafromBin->hiddenSize = 64;
    tilingDatafromBin->numOutTokens = 10;
    tilingDatafromBin->formerCoreNum = 10;
    tilingDatafromBin->tailCoreNum = 38;
    tilingDatafromBin->tokenNumEachCore = 1;
    tilingDatafromBin->tokenNumTailCore = 0;
    tilingDatafromBin->rowIdMapEachCore = 1;
    tilingDatafromBin->rowIdMapTailCore = 0;
    tilingDatafromBin->hiddenSizeAlign = 64;
    tilingDatafromBin->hiddenSizeLoopTimes = 1;
    tilingDatafromBin->hiddenSizeTail = 64;
    tilingDatafromBin->inputReserveNum = 1;
    tilingDatafromBin->indicesReserveNum = 1;
    tilingDatafromBin->indicesReserveNumAlign = 16;
    tilingDatafromBin->totalUbSize = 196352;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_bf16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 bfloat16_t bfloat16_t True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(bfloat16_t);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(bfloat16_t);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(bfloat16_t);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(bfloat16_t);
    size_t probsGradByteSize = 10 * 3 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_bf16_fp16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 bfloat16_t float16 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(bfloat16_t);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(bfloat16_t);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(half);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(bfloat16_t);
    size_t probsGradByteSize = 10 * 3 * sizeof(half);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_fp16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 float16 float16 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(half);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(half);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(half);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(half);
    size_t probsGradByteSize = 10 * 3 * sizeof(half);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_fp16_fp32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 float16 float32 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(half);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(half);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(float);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(half);
    size_t probsGradByteSize = 10 * 3 * sizeof(float);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_fp32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 float32 float32 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(float);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(float);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(float);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(float);
    size_t probsGradByteSize = 10 * 3 * sizeof(float);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_grad_test, test_case_prob_not_none_split_h_fp32_fp16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/moe_token_unpermute_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // token_num, topk, hiddensize, dtype, flag
    system("cd ./gen_data/ && python3 gen_data.py 10 3 8192 float32 float32 True");
    system("cd ./moe_token_unpermute_grad_data/ && python3 gen_tiling.py case2");

    size_t permutedTokensByteSize = 10 * 3 * 8192 * sizeof(float);
    size_t unpermutedOutputDByteSize = 10 * 8192 * sizeof(float);
    size_t sortedIndicesByteSize = 10 * 3 * sizeof(int32_t);
    size_t probsByteSize = 10 * 3 * sizeof(half);
    // output
    size_t permutedTokensGradByteSize = 10 * 3 * 8192 * sizeof(float);
    size_t probsGradByteSize = 10 * 3 * sizeof(half);
    size_t tilingDataSize = sizeof(MoeTokenUnpermuteGradTilingData);

    uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* unpermuted_output_d = (uint8_t*)AscendC::GmAlloc(unpermutedOutputDByteSize);
    uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
    uint8_t* permuted_tokens_grad = (uint8_t*)AscendC::GmAlloc(permutedTokensGradByteSize);
    uint8_t* probs_grad = (uint8_t*)AscendC::GmAlloc(probsGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_grad, blockDim, permuted_tokens, unpermuted_output_d, sorted_indices, probs,
        permuted_tokens_grad, probs_grad, workspace, tiling);

    AscendC::GmFree(permuted_tokens);
    AscendC::GmFree(unpermuted_output_d);
    AscendC::GmFree(sorted_indices);
    AscendC::GmFree(probs);
    AscendC::GmFree(permuted_tokens_grad);
    AscendC::GmFree(probs_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}