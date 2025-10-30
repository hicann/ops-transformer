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
#include <cmath>
#include "gtest/gtest.h"
#include "moe_init_routing_v2_grad_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(uint8_t *gradExpandedX, uint8_t *expandedRowIdx, uint8_t *gradX, uint8_t *workspace, uint8_t *tiling);

class MoeInitRoutingV2GradTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "MoeInitRoutingV2GradTest SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "MoeInitRoutingV2GradTest TearDown\n" << endl;
    }
};

// Dropless float
//            num_rows:   topk:   hidden_size:
//  case0:    8           2       5120

TEST_F(MoeInitRoutingV2GradTest, TestCase0)
{
    size_t num_rows = 8;
    size_t k = 2;
    size_t cols = 5120;
    uint64_t tilingKey = 1000;
    uint32_t blockDim = 1;

    size_t gradExpandedXFileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdxFileSize = num_rows * k * sizeof(int32_t);
    size_t gradXFileSize = num_rows * cols * sizeof(float);
    size_t workspaceFileSize = 16781184;
    size_t tilingFileSize = 22 * sizeof(int64_t);

    uint8_t *gradExpandedX = (uint8_t *)AscendC::GmAlloc(gradExpandedXFileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdxFileSize);
    uint8_t *gradX = (uint8_t *)AscendC::GmAlloc(gradXFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

    system("cp -r ../../../../../moe/moe_init_routing_v2_grad/tests/ut/op_kernel/moe_init_routing_v2_grad_data ./");
    system("chmod -R 755 ./moe_init_routing_v2_grad_data/");
    system("cd ./moe_init_routing_v2_grad_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_data.py 8 2 5120 0 0 0 0 float32");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_tiling.py case0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v2_grad_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v2_grad, blockDim, gradExpandedX, expandedRowIdx, gradX, workspace, tiling);

    AscendC::GmFree((void *)gradExpandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)gradX);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// Dropless float16
//            num_rows:   topk:   hidden_size:
//  case1:    4096        40      8
TEST_F(MoeInitRoutingV2GradTest, TestCase1)
{
    size_t num_rows = 4096;
    size_t k = 40;
    size_t cols = 8;
    uint64_t tilingKey = 1001;
    uint32_t blockDim = 1;

    size_t gradExpandedXFileSize = num_rows * k * cols * 2;
    size_t expandedRowIdxFileSize = num_rows * k * sizeof(int32_t);
    size_t gradXFileSize = num_rows * cols * 2;
    size_t workspaceFileSize = 16781184;
    size_t tilingFileSize = 22 * sizeof(int64_t);

    uint8_t *gradExpandedX = (uint8_t *)AscendC::GmAlloc(gradExpandedXFileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdxFileSize);
    uint8_t *gradX = (uint8_t *)AscendC::GmAlloc(gradXFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

    system("cp -r ../../../../../moe/moe_init_routing_v2_grad/tests/ut/op_kernel/moe_init_routing_v2_grad_data ./");
    system("chmod -R 755 ./moe_init_routing_v2_grad_data/");
    system("cd ./moe_init_routing_v2_grad_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_data.py 4096 40 8 0 0 0 0 float16");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_tiling.py case1");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v2_grad_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v2_grad, blockDim, gradExpandedX, expandedRowIdx, gradX, workspace, tiling);

    AscendC::GmFree((void *)gradExpandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)gradX);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// Active float16
//            num_rows:   topk:   hidden_size:   drop_pad_mode:   active_num:
//  case2:    10          64      512            0                40
TEST_F(MoeInitRoutingV2GradTest, TestCase2)
{
    size_t num_rows = 10;
    size_t k = 64;
    size_t cols = 512;
    size_t active_num = 40;
    uint64_t tilingKey = 1011;
    uint32_t blockDim = 1;

    size_t gradExpandedXFileSize = active_num * cols * 2;
    size_t expandedRowIdxFileSize = num_rows * k * sizeof(int32_t);
    size_t gradXFileSize = num_rows * cols * 2;
    size_t workspaceFileSize = 16781184;
    size_t tilingFileSize = 22 * sizeof(int64_t);

    uint8_t *gradExpandedX = (uint8_t *)AscendC::GmAlloc(gradExpandedXFileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdxFileSize);
    uint8_t *gradX = (uint8_t *)AscendC::GmAlloc(gradXFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

    system("cp -r ../../../../../moe/moe_init_routing_v2_grad/tests/ut/op_kernel/moe_init_routing_v2_grad_data ./");
    system("chmod -R 755 ./moe_init_routing_v2_grad_data/");
    system("cd ./moe_init_routing_v2_grad_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_data.py 10 64 512 0 0 0 40 float16");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v2_grad_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v2_grad, blockDim, gradExpandedX, expandedRowIdx, gradX, workspace, tiling);

    AscendC::GmFree((void *)gradExpandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)gradX);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// Drop/Pad bfloat16
//            num_rows:   topk:   hidden_size:   e:   c:   drop_pad_mode:   active_num:
//  case3:    80          8       512            10   8   1                40
TEST_F(MoeInitRoutingV2GradTest, TestCase3)
{
    size_t num_rows = 80;
    size_t k = 8;
    size_t cols = 512;
    size_t e = 10;
    size_t c = 8;
    uint64_t tilingKey = 1102;
    uint32_t blockDim = 1;

    size_t gradExpandedXFileSize = e * c * cols * 2;
    size_t expandedRowIdxFileSize = num_rows * k * sizeof(int32_t);
    size_t gradXFileSize = num_rows * cols * 2;
    size_t workspaceFileSize = 16781184;
    size_t tilingFileSize = 22 * sizeof(int64_t);

    uint8_t *gradExpandedX = (uint8_t *)AscendC::GmAlloc(gradExpandedXFileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdxFileSize);
    uint8_t *gradX = (uint8_t *)AscendC::GmAlloc(gradXFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

    system("cp -r ../../../../../moe/moe_init_routing_v2_grad/tests/ut/op_kernel/moe_init_routing_v2_grad_data ./");
    system("chmod -R 755 ./moe_init_routing_v2_grad_data/");
    system("cd ./moe_init_routing_v2_grad_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_data.py 80 8 512 10 8 1 40 bfloat16");
    system("cd ./moe_init_routing_v2_grad_data/ && python3 gen_tiling.py case3");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v2_grad_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v2_grad, blockDim, gradExpandedX, expandedRowIdx, gradX, workspace, tiling);

    AscendC::GmFree((void *)gradExpandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)gradX);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}