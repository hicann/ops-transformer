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
#include "gtest/gtest.h"
#include "moe_init_routing_quant_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_quant_v2(
    uint8_t* x, uint8_t* expertIdx, uint8_t* scale, uint8_t* offset, uint8_t* expandedX, uint8_t* expandedRowIdx,
    uint8_t* expertTokensCountOrCumsum, uint8_t* expertTokensBeforeCapacity, uint8_t* dynamicQuantScale,
    uint8_t* workspace, uint8_t* tiling);

class moe_init_routing_quant_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_init_routing_quant_v2_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_init_routing_quant_v2_test TearDown\n" << endl;
  }
};

TEST_F(moe_init_routing_quant_v2_test, test_case_0) {
  size_t num_rows = 80;
  size_t k = 60;
  size_t cols = 3000;
  size_t active_rows = 0;
  size_t capacity_num = 6;
  size_t expert_num = 8;
  size_t drop_pad_mode = 0;
  size_t expert_tokens_count_or_cumsum = 1;
  bool expert_tokens_before_capacity = false;
  size_t quant_mode = 0;
  uint64_t tilingKey = 20000;
  uint32_t blockDim = 32;

  size_t x_FileSize = num_rows * cols * sizeof(float);
  size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t scale_FileSize = sizeof(float);
  size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
  size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expertTokensCumsum_FileSize = expert_num * sizeof(int32_t);
  size_t workspace_FileSize = num_rows * k * sizeof(float) * 8 + blockDim * sizeof(int32_t) * 2 + 16781184;
  size_t tiling_FileSize = sizeof(MoeInitRoutingQuantV2TilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
  uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
  uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_FileSize);
  uint8_t* offset = (uint8_t*)AscendC::GmAlloc(scale_FileSize);
  uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
  uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdx_FileSize);
  uint8_t* expertTokensCountOrCumsum = (uint8_t*)AscendC::GmAlloc(expertTokensCumsum_FileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);
  system("cp -r ../../../../../moe/moe_init_routing_quant_v2/tests/ut/op_kernel/moe_init_routing_quant_v2_data ./");
  system("chmod -R 755 ./moe_init_routing_quant_v2_data/");
  system("cd ./moe_init_routing_quant_v2_data/ && rm -rf ./*bin");
  system("cd ./moe_init_routing_quant_v2_data/ && python3 gen_data.py 80 60 3000 0 6 8 0 1 false 0 float32");
  system("cd ./moe_init_routing_quant_v2_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "/moe_init_routing_quant_v2_data/input_x.bin", x_FileSize, x, x_FileSize);
  ReadFile(path + "/moe_init_routing_quant_v2_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx,
           expertIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_v2_data/scale.bin", scale_FileSize, scale, scale_FileSize);
  ReadFile(path + "/moe_init_routing_quant_v2_data/offset.bin", scale_FileSize, offset, scale_FileSize);
  ReadFile(path + "/moe_init_routing_quant_v2_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(moe_init_routing_quant_v2, blockDim, x, expertIdx, scale, offset, expandedX, expandedRowIdx,
              expertTokensCountOrCumsum, nullptr, nullptr, workspace, tiling);
  // WriteFile(path + "/moe_init_routing_data/out_0.bin", expandedX, expandedX_FileSize);
  // WriteFile(path + "/moe_init_routing_data/out_1.bin", expandedRowIdx, expandedRowIdx_FileSize);
  // WriteFile(path + "/moe_init_routing_data/out_2.bin", expandedExpertIdx, expandedExpertIdx_FileSize);

  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)expertIdx);
  AscendC::GmFree((void*)scale);
  AscendC::GmFree((void*)offset);
  AscendC::GmFree((void*)expandedX);
  AscendC::GmFree((void*)expandedRowIdx);
  AscendC::GmFree((void*)expertTokensCountOrCumsum);
  AscendC::GmFree((void*)workspace);
  AscendC::GmFree((void*)tiling);
  free(path_);
}