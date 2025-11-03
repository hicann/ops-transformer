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
 * \file test_moe_token_unpermute.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_moe_token_unpermute.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_unpermute(GM_ADDR permuted_tokens, GM_ADDR sorted_indices,
                                                          GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR workspace,
                                                          GM_ADDR tiling);

class moe_token_unpermute_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_token_unpermute_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_token_unpermute_test TearDown\n" << endl;
  }
};

TEST_F(moe_token_unpermute_test, test_prob_none_bf16) {
  size_t permutedTokensByteSize = 90 * 8 * 1024 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 90 * 8 * sizeof(int32_t);
  size_t probsByteSize = 90 * 8 * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = 90 * 1024 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, nullptr, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_bf16) {
  size_t permutedTokensByteSize = 6144 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  size_t probsByteSize = 6144 * 8 * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = 6144 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_prob_none_fp16) {
  size_t permutedTokensByteSize = 72 * 8 * 5120 * sizeof(half);
  size_t sortedIndicesByteSize = 72 * 8 * sizeof(int32_t);
  size_t probsByteSize = 72 * 8 * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = 72 * 5120 * sizeof(half);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(2);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, nullptr, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_fp16) {
  size_t permutedTokensByteSize = 32 * 8 * 5122 * sizeof(half);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = 32 * 5122 * sizeof(half);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(3);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_prob_none_fp32) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(float);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(float);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(4);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, nullptr, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_fp32) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(float);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(float);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(5);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_test, test_mix_bf16_fp16) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(17);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_mix_bf16_fp32) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(25);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_mix_fp16_bf16) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(half);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(half);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(11);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_mix_fp16_fp32) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(half);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(half);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(27);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_mix_fp32_bf16) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(float);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(float);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(13);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_test, test_mix_fp32_fp16) {
  size_t permutedTokensByteSize = 32 * 8 * 5120 * sizeof(float);
  size_t sortedIndicesByteSize = 32 * 8 * sizeof(int32_t);
  size_t probsByteSize = 32 * 8 * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = 32 * 5120 * sizeof(float);
  size_t tilingDataSize = sizeof(MoeTokenUnpermuteTilingData);

  uint8_t* permuted_tokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* unpermuted_tokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteTilingData*>(tiling);
  tilingDatafromBin->buffer_num=1;
  tilingDatafromBin->hidden_splited_length=1;
  tilingDatafromBin->tokens_splited_length=1;
  tilingDatafromBin->top_k=1;
  ICPU_SET_TILING_KEY(21);
  ICPU_RUN_KF(moe_token_unpermute, blockDim, permuted_tokens, sorted_indices, probs, unpermuted_tokens, workspace,
              (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permuted_tokens);
  AscendC::GmFree(sorted_indices);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermuted_tokens);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}
