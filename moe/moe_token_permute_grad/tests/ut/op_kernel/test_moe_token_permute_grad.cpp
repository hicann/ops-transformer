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
#include "test_moe_token_permute_grad.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_permute_grad(GM_ADDR permuted_output_d, GM_ADDR sorted_indices,
                                                             GM_ADDR input_grad, GM_ADDR workspace, GM_ADDR tiling);
class moe_token_permute_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_token_permute_grad_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_token_permute_grad_test TearDown\n" << endl;
  }
};

TEST_F(moe_token_permute_grad_test, test_bf16) {

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t permutedOutDByteSize = 6144 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  // size_t probsByteSize = 6144 * 8 * sizeof(bfloat16_t);
  // output
  size_t inputDByteSize = 6144 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenPermuteGradTilingData);

  uint8_t* permuted_output_d = (uint8_t*)AscendC::GmAlloc(permutedOutDByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  // uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* input_grad = (uint8_t*)AscendC::GmAlloc(inputDByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteGradTilingData* tilingData = reinterpret_cast<MoeTokenPermuteGradTilingData*>(tiling);
  tilingData->hidden_size = 5120;
  tilingData->top_k = 8;
  tilingData->num_out_tokens = 49152;
  tilingData->hidden_splited_length = 5120;
  tilingData->hidden_splited_num = 1;
  tilingData->hidden_splited_remain = 0;
  tilingData->tokens_core_length = 96;
  tilingData->tokens_core_remain = 0;
  tilingData->tokens_splited_length = 96;
  tilingData->tokens_splited_num = 1;
  tilingData->tokens_splited_remain = 0;
  tilingData->buffer_num = 4;
  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_permute_grad, blockDim, permuted_output_d, sorted_indices, input_grad, workspace, tiling);

  AscendC::GmFree(permuted_output_d);
  AscendC::GmFree(sorted_indices);
  // AscendC::GmFree(probs);
  AscendC::GmFree(input_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_permute_grad_test, test_fp16) {
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t permutedOutDByteSize = 6144 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  // size_t probsByteSize = 6144 * 8 * sizeof(bfloat16_t);
  // output
  size_t inputDByteSize = 6144 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenPermuteGradTilingData);

  uint8_t* permuted_output_d = (uint8_t*)AscendC::GmAlloc(permutedOutDByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  // uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* input_grad = (uint8_t*)AscendC::GmAlloc(inputDByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteGradTilingData* tilingData = reinterpret_cast<MoeTokenPermuteGradTilingData*>(tiling);
  tilingData->hidden_size = 5120;
  tilingData->top_k = 8;
  tilingData->num_out_tokens = 49152;
  tilingData->hidden_splited_length = 5120;
  tilingData->hidden_splited_num = 1;
  tilingData->hidden_splited_remain = 0;
  tilingData->tokens_core_length = 96;
  tilingData->tokens_core_remain = 0;
  tilingData->tokens_splited_length = 96;
  tilingData->tokens_splited_num = 1;
  tilingData->tokens_splited_remain = 0;
  tilingData->buffer_num = 4;
  ICPU_SET_TILING_KEY(2);
  ICPU_RUN_KF(moe_token_permute_grad, blockDim, permuted_output_d, sorted_indices, input_grad, workspace, tiling);

  AscendC::GmFree(permuted_output_d);
  AscendC::GmFree(sorted_indices);
  // AscendC::GmFree(probs);
  AscendC::GmFree(input_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_permute_grad_test, test_fp32) {

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t permutedOutDByteSize = 6144 * 8 * 5120 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  // size_t probsByteSize = 6144 * 8 * sizeof(bfloat16_t);
  // output
  size_t inputDByteSize = 6144 * 5120 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(MoeTokenPermuteGradTilingData);

  uint8_t* permuted_output_d = (uint8_t*)AscendC::GmAlloc(permutedOutDByteSize + 32);
  uint8_t* sorted_indices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize + 32);
  // uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize + 32);

  uint8_t* input_grad = (uint8_t*)AscendC::GmAlloc(inputDByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteGradTilingData* tilingData = reinterpret_cast<MoeTokenPermuteGradTilingData*>(tiling);
  tilingData->hidden_size = 5120;
  tilingData->top_k = 8;
  tilingData->num_out_tokens = 49152;
  tilingData->hidden_splited_length = 5120;
  tilingData->hidden_splited_num = 1;
  tilingData->hidden_splited_remain = 0;
  tilingData->tokens_core_length = 96;
  tilingData->tokens_core_remain = 0;
  tilingData->tokens_splited_length = 96;
  tilingData->tokens_splited_num = 1;
  tilingData->tokens_splited_remain = 0;
  tilingData->buffer_num = 4;

  ICPU_SET_TILING_KEY(4);
  ICPU_RUN_KF(moe_token_permute_grad, blockDim, permuted_output_d, sorted_indices, input_grad, workspace, tiling);

  AscendC::GmFree(permuted_output_d);
  AscendC::GmFree(sorted_indices);
  // AscendC::GmFree(probs);
  AscendC::GmFree(input_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}