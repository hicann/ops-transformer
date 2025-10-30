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
 * \file test_moe_init_routing_quant.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "moe_init_routing_quant_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_quant(uint8_t* x, uint8_t* rowIdx, uint8_t* expertIdx,
                                                             uint8_t* expandedX, uint8_t* expandedRowIdx,
                                                             uint8_t* expandedExpertIdx, uint8_t* workspace,
                                                             uint8_t* tiling);

class moe_init_routing_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_init_routing_quant_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_init_routing_quant_test TearDown\n" << endl;
  }
};

//            num_rows:   topk:   hidden_size:    active_rows:
//  case0:    8           2       5120            5120
//  case1:    4096        40      8               8
//  case2:    1           48      1024            1
//  case3:    20          20      44321           20

TEST_F(moe_init_routing_quant_test, test_case_0) {
  size_t num_rows = 8;
  size_t k = 2;
  size_t cols = 5120;
  size_t active_rows = 8;
  uint64_t tilingKey = 1;
  uint32_t blockDim = 1;

  size_t out_row = num_rows > active_rows ? active_rows : num_rows;

  size_t x_FileSize = num_rows * cols * sizeof(float);
  size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t rowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
  size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t workspace_FileSize =
      num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
  size_t tiling_FileSize = 48 * sizeof(int64_t) + 2 * sizeof(float);

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
  uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
  uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
  uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
  uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdx_FileSize);
  uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);

  system(
      "cp -r "
      "../../../../../moe/moe_init_routing_quant/tests/ut/op_kernel/moe_init_routing_quant_data ./");
  system("chmod -R 755 ./moe_init_routing_quant_data/");
  system("cd ./moe_init_routing_quant_data/ && rm -rf ./*bin");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_data.py 8 2 5120 8 0 0 float32");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_tiling.py case0");
  char* path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "/moe_init_routing_quant_data/input_x.bin", x_FileSize, x, x_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx,
           expertIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_rowIdx.bin", rowIdx_FileSize, rowIdx, rowIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(moe_init_routing_quant, blockDim, x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx,
              workspace, tiling);
  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)expertIdx);
  AscendC::GmFree((void*)rowIdx);
  AscendC::GmFree((void*)expandedX);
  AscendC::GmFree((void*)expandedRowIdx);
  AscendC::GmFree((void*)expandedExpertIdx);
  AscendC::GmFree((void*)workspace);
  AscendC::GmFree((void*)tiling);
  free(path_);
}

TEST_F(moe_init_routing_quant_test, test_case_1) {
  size_t num_rows = 8;
  size_t k = 2;
  size_t cols = 5120;
  size_t active_rows = 8;
  uint64_t tilingKey = 2;
  uint32_t blockDim = 1;

  size_t out_row = num_rows > active_rows ? active_rows : num_rows;

  size_t x_FileSize = num_rows * cols * sizeof(float);
  size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t rowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
  size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t workspace_FileSize =
      num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
  size_t tiling_FileSize = 48 * sizeof(int64_t) + 2 * sizeof(float);

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
  uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
  uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
  uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
  uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdx_FileSize);
  uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);

  system(
      "cp -r "
      "../../../../../moe/moe_init_routing_quant/tests/ut/op_kernel/moe_init_routing_quant_data ./");
  system("chmod -R 755 ./moe_init_routing_quant_data/");
  system("cd ./moe_init_routing_quant_data/ && rm -rf ./*bin");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_data.py 8 2 5120 8 0 0 float32");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "/moe_init_routing_quant_data/input_x.bin", x_FileSize, x, x_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx,
           expertIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_rowIdx.bin", rowIdx_FileSize, rowIdx, rowIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(moe_init_routing_quant, blockDim, x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx,
              workspace, tiling);

  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)expertIdx);
  AscendC::GmFree((void*)rowIdx);
  AscendC::GmFree((void*)expandedX);
  AscendC::GmFree((void*)expandedRowIdx);
  AscendC::GmFree((void*)expandedExpertIdx);
  AscendC::GmFree((void*)workspace);
  AscendC::GmFree((void*)tiling);
  free(path_);
}

TEST_F(moe_init_routing_quant_test, test_case_2) {
  size_t num_rows = 8;
  size_t k = 2;
  size_t cols = 5120;
  size_t active_rows = 8;
  uint64_t tilingKey = 3;
  uint32_t blockDim = 1;

  size_t out_row = num_rows > active_rows ? active_rows : num_rows;

  size_t x_FileSize = num_rows * cols * sizeof(float);
  size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t rowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
  size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t workspace_FileSize =
      num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
  size_t tiling_FileSize = 48 * sizeof(int64_t) + 2 * sizeof(float);

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
  uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
  uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
  uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
  uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdx_FileSize);
  uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);

  system(
      "cp -r "
      "../../../../../moe/moe_init_routing_quant/tests/ut/op_kernel/moe_init_routing_quant_data ./");
  system("chmod -R 755 ./moe_init_routing_quant_data/");
  system("cd ./moe_init_routing_quant_data/ && rm -rf ./*bin");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_data.py 8 2 5120 8 0 0 float32");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "/moe_init_routing_quant_data/input_x.bin", x_FileSize, x, x_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx,
           expertIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_rowIdx.bin", rowIdx_FileSize, rowIdx, rowIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(moe_init_routing_quant, blockDim, x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx,
              workspace, tiling);

  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)expertIdx);
  AscendC::GmFree((void*)rowIdx);
  AscendC::GmFree((void*)expandedX);
  AscendC::GmFree((void*)expandedRowIdx);
  AscendC::GmFree((void*)expandedExpertIdx);
  AscendC::GmFree((void*)workspace);
  AscendC::GmFree((void*)tiling);
  free(path_);
}

TEST_F(moe_init_routing_quant_test, test_case_3) {
  size_t num_rows = 8;
  size_t k = 2;
  size_t cols = 5120;
  size_t active_rows = 8;
  uint64_t tilingKey = 4;
  uint32_t blockDim = 1;

  size_t out_row = num_rows > active_rows ? active_rows : num_rows;

  size_t x_FileSize = num_rows * cols * sizeof(float);
  size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t rowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedX_FileSize = out_row * k * cols * sizeof(float);
  size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t expandedExpertIdx_FileSize = num_rows * k * sizeof(int32_t);
  size_t workspace_FileSize =
      num_rows * k * sizeof(float) * 2 * 3 + blockDim * 32 * 2 + num_rows * k * sizeof(int32_t) + 16781184;
  size_t tiling_FileSize = 48 * sizeof(int64_t) + 2 * sizeof(float);

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
  uint8_t* expertIdx = (uint8_t*)AscendC::GmAlloc(expertIdx_FileSize);
  uint8_t* rowIdx = (uint8_t*)AscendC::GmAlloc(rowIdx_FileSize);
  uint8_t* expandedX = (uint8_t*)AscendC::GmAlloc(expandedX_FileSize);
  uint8_t* expandedRowIdx = (uint8_t*)AscendC::GmAlloc(expandedRowIdx_FileSize);
  uint8_t* expandedExpertIdx = (uint8_t*)AscendC::GmAlloc(expandedExpertIdx_FileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);

  system(
      "cp -r "
      "../../../../../moe/moe_init_routing_quant/tests/ut/op_kernel/moe_init_routing_quant_data ./");
  system("chmod -R 755 ./moe_init_routing_quant_data/");
  system("cd ./moe_init_routing_quant_data/ && rm -rf ./*bin");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_data.py 8 2 5120 8 0 0 float32");
  system("cd ./moe_init_routing_quant_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "/moe_init_routing_quant_data/input_x.bin", x_FileSize, x, x_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx,
           expertIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/input_rowIdx.bin", rowIdx_FileSize, rowIdx, rowIdx_FileSize);
  ReadFile(path + "/moe_init_routing_quant_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(moe_init_routing_quant, blockDim, x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx,
              workspace, tiling);
  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)expertIdx);
  AscendC::GmFree((void*)rowIdx);
  AscendC::GmFree((void*)expandedX);
  AscendC::GmFree((void*)expandedRowIdx);
  AscendC::GmFree((void*)expandedExpertIdx);
  AscendC::GmFree((void*)workspace);
  AscendC::GmFree((void*)tiling);
  free(path_);
}
