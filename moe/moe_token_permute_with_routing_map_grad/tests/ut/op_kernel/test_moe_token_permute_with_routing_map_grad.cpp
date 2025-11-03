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
#include "test_moe_token_permute_with_routing_map_grad.h"
// #include "../data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map_grad(GM_ADDR permutedTokenOutPutGrad,
                                                                              GM_ADDR permutedProbsOutPutGradOptional,
                                                                              GM_ADDR sortedIndices,
                                                                              GM_ADDR routingMapOptional,
                                                                              GM_ADDR tokensGradOut,
                                                                              GM_ADDR probsGradOutOptional,
                                                                              GM_ADDR workspace, GM_ADDR tiling);

class moe_token_permute_with_routing_map_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_token_permute_with_routing_map_grad_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_token_permute_with_routing_map_grad_test TearDown\n" << endl;
  }
};

TEST_F(moe_token_permute_with_routing_map_grad_test, test_bf16) {

  size_t permutedTokenOutPutGradByteSize = 6144 * 8 * 5120 * sizeof(bfloat16_t);
  size_t permutedProbsOutPutGradOptionalByteSize = 6144 * 8 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 6144 * 8;
  // output
  size_t tokensGradOutByteSize = 6144 * 5120 * sizeof(bfloat16_t);
  size_t probsGradOutOptionalByteSize = 6144 * 5120 * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_permute_with_routing_map_grad_test, test_fp16) {

  size_t permutedTokenOutPutGradByteSize = 6144 * 8 * 5120 * sizeof(half);
  size_t permutedProbsOutPutGradOptionalByteSize = 6144 * 8 * sizeof(half);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 6144 * 8;
  // output
  size_t tokensGradOutByteSize = 6144 * 5120 * sizeof(half);
  size_t probsGradOutOptionalByteSize = 6144 * 5120 * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(2);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_permute_with_routing_map_grad_test, test_fp32) {

  size_t permutedTokenOutPutGradByteSize = 6144 * 8 * 5120 * sizeof(float);
  size_t permutedProbsOutPutGradOptionalByteSize = 6144 * 8 * sizeof(float);
  size_t sortedIndicesByteSize = 6144 * 8 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 6144 * 8;
  // output
  size_t tokensGradOutByteSize = 6144 * 5120 * sizeof(float);
  size_t probsGradOutOptionalByteSize = 6144 * 5120 * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(4);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}



TEST_F(moe_token_permute_with_routing_map_grad_test, test_bf16_droppad) {

  size_t permutedTokenOutPutGradByteSize = 1024 * 7168 * sizeof(bfloat16_t);
  size_t permutedProbsOutPutGradOptionalByteSize = 1024 * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = 1024 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 512 * 512;
  // output
  size_t tokensGradOutByteSize = 512 * 7168 * sizeof(bfloat16_t);
  size_t probsGradOutOptionalByteSize = 512 * 512 * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_permute_with_routing_map_grad_test, test_fp16_droppad) {

  size_t permutedTokenOutPutGradByteSize = 1024 * 7168 * sizeof(half);
  size_t permutedProbsOutPutGradOptionalByteSize = 1024 * sizeof(half);
  size_t sortedIndicesByteSize = 1024 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 512 * 512;
  // output
  size_t tokensGradOutByteSize = 512 * 7168 * sizeof(half);
  size_t probsGradOutOptionalByteSize = 512 * 512 * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(1002);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_permute_with_routing_map_grad_test, test_fp32_droppad) {

  size_t permutedTokenOutPutGradByteSize = 1024 * 7168 * sizeof(float);
  size_t permutedProbsOutPutGradOptionalByteSize = 1024 * sizeof(float);
  size_t sortedIndicesByteSize = 1024 * sizeof(int32_t);
  size_t routingMapOptionalByteSize = 512 * 512;
  // output
  size_t tokensGradOutByteSize = 512 * 7168 * sizeof(float);
  size_t probsGradOutOptionalByteSize = 512 * 512 * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenPermuteWithRoutingMapGradTilingData);

  uint8_t* permutedTokenOutPutGrad = (uint8_t*)AscendC::GmAlloc(permutedTokenOutPutGradByteSize);
  uint8_t* permutedProbsOutPutGradOptional = (uint8_t*)AscendC::GmAlloc(permutedProbsOutPutGradOptionalByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMapOptional = (uint8_t*)AscendC::GmAlloc(routingMapOptionalByteSize);
  uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
  uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenPermuteWithRoutingMapGradTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenPermuteWithRoutingMapGradTilingData*>(tiling);
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.buffer_num=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.hidden_splited_length=1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.top_k = 1;
  tilingDatafromBin->moeTokenPermuteWithRoutingMapGradUnpermuteTilingData.tokens_splited_length = 1;
  ICPU_SET_TILING_KEY(1004);
  ICPU_RUN_KF(moe_token_permute_with_routing_map_grad, blockDim, permutedTokenOutPutGrad, permutedProbsOutPutGradOptional,
              sortedIndices, routingMapOptional, tokensGradOut, probsGradOutOptional, workspace, tiling);

  AscendC::GmFree(permutedTokenOutPutGrad);
  AscendC::GmFree(permutedProbsOutPutGradOptional);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMapOptional);
  AscendC::GmFree(tokensGradOut);
  AscendC::GmFree(probsGradOutOptional);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}
