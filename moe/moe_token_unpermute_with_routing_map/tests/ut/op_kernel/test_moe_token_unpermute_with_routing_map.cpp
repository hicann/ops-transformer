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
#include "moe_token_unpermute_with_routing_map_tiling.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map(GM_ADDR permutedTokens,
                                                                           GM_ADDR sortedIndices,
                                                                           GM_ADDR routingMap,
                                                                           GM_ADDR probs,
                                                                           GM_ADDR unpermutedTokens,
                                                                           GM_ADDR outIndex,
                                                                           GM_ADDR permuteTokenId,
                                                                           GM_ADDR permuteProbs,
                                                                           GM_ADDR workspace, GM_ADDR tiling);

class moe_token_unpermute_with_routing_map_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_token_unpermute_with_routing_map_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_token_unpermute_with_routing_map_test TearDown\n" << endl;
  }
};

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 8;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 8;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 64;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 512;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 1;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp32) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 8;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 8;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 64;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 512;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 1;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_bf16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 8;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 8;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 64;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 1;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 512;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 512;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 1;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp16_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 64;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 64;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 4;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 64;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 458752;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 112;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 4096;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 4;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp32_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 64;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 64;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 4;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 64;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 458752;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 112;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 4096;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 4;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_bf16_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 7168;
  tilingDatafromBin->top_k = 8;
  tilingDatafromBin->num_out_tokens = 32768;
  tilingDatafromBin->hidden_splited_length = 7168;
  tilingDatafromBin->hidden_splited_num = 1;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 64;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 64;
  tilingDatafromBin->tokens_splited_num = 1;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 4;
  tilingDatafromBin->buffer_num = 4;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 64;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 458752;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 112;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 4096;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 4096;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 4;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 0;

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_fp32) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 0;
  tilingDatafromBin->top_k = 0;
  tilingDatafromBin->num_out_tokens = 0;
  tilingDatafromBin->hidden_splited_length = 0;
  tilingDatafromBin->hidden_splited_num = 0;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 0;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 0;
  tilingDatafromBin->tokens_splited_num = 0;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 0;
  tilingDatafromBin->buffer_num = 0;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 0;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 4096;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 256;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 4;

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_fp16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);
  
  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 0;
  tilingDatafromBin->top_k = 0;
  tilingDatafromBin->num_out_tokens = 0;
  tilingDatafromBin->hidden_splited_length = 0;
  tilingDatafromBin->hidden_splited_num = 0;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 0;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 0;
  tilingDatafromBin->tokens_splited_num = 0;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 0;
  tilingDatafromBin->buffer_num = 0;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 0;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 4096;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 256;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 4;

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_bf16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  tilingDatafromBin->hidden_size = 0;
  tilingDatafromBin->top_k = 0;
  tilingDatafromBin->num_out_tokens = 0;
  tilingDatafromBin->hidden_splited_length = 0;
  tilingDatafromBin->hidden_splited_num = 0;
  tilingDatafromBin->hidden_splited_remain = 0;
  tilingDatafromBin->tokens_core_length = 0;
  tilingDatafromBin->tokens_core_remain = 0;
  tilingDatafromBin->tokens_splited_length = 0;
  tilingDatafromBin->tokens_splited_num = 0;
  tilingDatafromBin->tokens_splited_remain = 0;
  tilingDatafromBin->used_core_num = 0;
  tilingDatafromBin->buffer_num = 0;

  // 嵌套结构体 maskedSelectParamsOp 的赋值
  tilingDatafromBin->maskedSelectParamsOp.formerNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.formertileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.formerlasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.tailtileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.taillasttileLength = 0;
  tilingDatafromBin->maskedSelectParamsOp.tokenNum = 0;
  tilingDatafromBin->maskedSelectParamsOp.needCoreNum = 0;

  // 嵌套结构体 moeTokenUnpermuteWithRoutingMapPadTilingData 的赋值
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.core_num_use = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens = 4096;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_experts = 256;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.capacity = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.front_core = 64;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_front_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_front_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_front_core_last_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.tail_core = 0;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.loop_time_each_tail_core = 1;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_each_tail_core = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_each_loop = 4;
  tilingDatafromBin->moeTokenUnpermuteWithRoutingMapPadTilingData.num_tokens_tail_core_last_loop = 4;

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, (uint8_t*)tilingDatafromBin);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}