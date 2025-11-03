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
#include "moe_init_routing_v3_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_v3(uint8_t *x, uint8_t *expertIdx, uint8_t *scale,
                                                          uint8_t *offset, uint8_t *expandedX, uint8_t *expandedRowIdx,
                                                          uint8_t *expertTokensCountOrCumsum, uint8_t *expandedScale,
                                                          uint8_t *workspace, uint8_t *tiling);

class moe_init_routing_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "moe_init_routing_v3_test SetUp\n" << endl;
        system(
            "cp -rf "
            "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/moe_init_routing_v3_data ./");
    }
    static void TearDownTestCase()
    {
        cout << "moe_init_routing_v3_test TearDown\n" << endl;
    }
};

// // 多核排序、非量化、GATHER索引
// TEST_F(moe_init_routing_v3_test, test_case_0)
// {
//     size_t num_rows = 160;
//     size_t cols = 96;
//     size_t k = 1450;
//     size_t expert_num = 12;
//     uint64_t tilingKey = 1100000;
//     uint32_t blockDim = 1;

//     size_t x_FileSize = num_rows * cols * sizeof(float);
//     size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t scale_FileSize = num_rows * sizeof(float);
//     size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
//     size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
//     size_t expandedScale_FileSize = num_rows * sizeof(float);
//     size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
//                                                     expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
//     size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

//     uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
//     uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
//     uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
//     uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
//     uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
//     uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
//     uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
//     uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

//     system("chmod -R 755 ./moe_init_routing_v3_data/");
//     system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 160 96 1450 12 float32");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case0");

//     std::string curPath = ".";
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tilingKey);
//     ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
//                 expertTokensCountOrCumsum, expandedScale, workspace, tiling);

//     AscendC::GmFree((void *)x);
//     AscendC::GmFree((void *)expertIdx);
//     AscendC::GmFree((void *)scale);
//     AscendC::GmFree((void *)expandedX);
//     AscendC::GmFree((void *)expandedRowIdx);
//     AscendC::GmFree((void *)expertTokensCountOrCumsum);
//     AscendC::GmFree((void *)expandedScale);
//     AscendC::GmFree((void *)workspace);
//     AscendC::GmFree((void *)tiling);
// }

// // 单核排序、动态量化、GATHER索引
// TEST_F(moe_init_routing_v3_test, test_case_1)
// {
//     size_t num_rows = 1;
//     size_t cols = 83;
//     size_t k = 27;
//     size_t expert_num = 12;
//     uint64_t tilingKey = 1020000;
//     uint32_t blockDim = 40;

//     size_t x_FileSize = num_rows * cols * sizeof(float);
//     size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t scale_FileSize = num_rows * sizeof(float);
//     size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
//     size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
//     size_t expandedScale_FileSize = num_rows * sizeof(float);
//     size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
//                                                     expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
//     size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

//     uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
//     uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
//     uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
//     uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
//     uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
//     uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
//     uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
//     uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

//     system("chmod -R 755 ./moe_init_routing_v3_data/");
//     system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 1 83 27 12 float32");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case1");

//     std::string curPath = ".";
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tilingKey);
//     ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
//                 expertTokensCountOrCumsum, expandedScale, workspace, tiling);

//     AscendC::GmFree((void *)x);
//     AscendC::GmFree((void *)expertIdx);
//     AscendC::GmFree((void *)scale);
//     AscendC::GmFree((void *)expandedX);
//     AscendC::GmFree((void *)expandedRowIdx);
//     AscendC::GmFree((void *)expertTokensCountOrCumsum);
//     AscendC::GmFree((void *)expandedScale);
//     AscendC::GmFree((void *)workspace);
//     AscendC::GmFree((void *)tiling);
// }

TEST_F(moe_init_routing_v3_test, test_case_2)
{
    size_t num_rows = 2730;
    size_t cols = 6144;
    size_t k = 8;
    size_t expert_num = 8; // end-start
    uint64_t tilingKey = 1201000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system("chmod -R 755 ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 2730 6144 8 256 int8");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case2");

    std::string curPath = ".";
    ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

// TEST_F(moe_init_routing_v3_test, test_case_4)
// {
//     size_t num_rows = 32;
//     size_t cols = 674;
//     size_t k = 5205;
//     size_t expert_num = 321; // [107, 428]
//     uint64_t tilingKey = 1100000;
//     uint32_t blockDim = 40;

//     size_t x_FileSize = num_rows * cols * sizeof(float);
//     size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t scale_FileSize = num_rows * sizeof(float);
//     size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
//     size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
//     size_t expandedScale_FileSize = num_rows * sizeof(float);
//     size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
//                                                     expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
//     size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

//     uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
//     uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
//     uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
//     uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
//     uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
//     uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
//     uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
//     uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

//     system("chmod -R 755 ./moe_init_routing_v3_data/");
//     system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 32 674 5205 1024 int8");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case4");

//     std::string curPath = ".";
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tilingKey);
//     ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
//                 expertTokensCountOrCumsum, expandedScale, workspace, tiling);

//     AscendC::GmFree((void *)x);
//     AscendC::GmFree((void *)expertIdx);
//     AscendC::GmFree((void *)scale);
//     AscendC::GmFree((void *)expandedX);
//     AscendC::GmFree((void *)expandedRowIdx);
//     AscendC::GmFree((void *)expertTokensCountOrCumsum);
//     AscendC::GmFree((void *)expandedScale);
//     AscendC::GmFree((void *)workspace);
//     AscendC::GmFree((void *)tiling);
// }

// TEST_F(moe_init_routing_v3_test, test_case_5)
// {
//     size_t num_rows = 35;
//     size_t cols = 2505;
//     size_t k = 8;
//     size_t expert_num = 620;
//     uint64_t tilingKey = 1020000;
//     uint32_t blockDim = 40;

//     size_t x_FileSize = num_rows * cols * sizeof(float);
//     size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
//     size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expandedScale_FileSize = num_rows * sizeof(float);
//     size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
//     size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
//                                                     expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
//     size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

//     uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
//     uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
//     uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
//     uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
//     uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
//     uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
//     uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

//     system("chmod -R 755 ./moe_init_routing_v3_data/");
//     system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 35 2505 8 620 float16");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case5");

//     std::string curPath = ".";
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tilingKey);
//     ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
//                 expertTokensCountOrCumsum, expandedScale, workspace, tiling);

//     AscendC::GmFree((void *)x);
//     AscendC::GmFree((void *)expertIdx);
//     AscendC::GmFree((void *)expandedX);
//     AscendC::GmFree((void *)expandedRowIdx);
//     AscendC::GmFree((void *)expertTokensCountOrCumsum);
//     AscendC::GmFree((void *)expandedScale);
//     AscendC::GmFree((void *)workspace);
//     AscendC::GmFree((void *)tiling);
// }

// TEST_F(moe_init_routing_v3_test, test_case_6)
// {
//     size_t num_rows = 1;
//     size_t cols = 7168;
//     size_t k = 8;
//     size_t expert_num = 256;
//     uint64_t tilingKey = 2000000;
//     uint32_t blockDim = 40;

//     size_t x_FileSize = num_rows * cols * sizeof(float);
//     size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t scale_FileSize = num_rows * sizeof(float);
//     size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
//     size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
//     size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
//     size_t expandedScale_FileSize = num_rows * sizeof(float);
//     size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
//                                                     expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
//     size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

//     uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
//     uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
//     uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
//     uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
//     uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
//     uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
//     uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
//     uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

//     system("chmod -R 755 ./moe_init_routing_v3_data/");
//     system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 1 7168 8 256 float16"); // bf16
//     system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case6");

//     std::string curPath = ".";
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
//     ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tilingKey);
//     ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
//                 expertTokensCountOrCumsum, expandedScale, workspace, tiling);

//     AscendC::GmFree((void *)x);
//     AscendC::GmFree((void *)expertIdx);
//     AscendC::GmFree((void *)scale);
//     AscendC::GmFree((void *)expandedX);
//     AscendC::GmFree((void *)expandedRowIdx);
//     AscendC::GmFree((void *)expertTokensCountOrCumsum);
//     AscendC::GmFree((void *)expandedScale);
//     AscendC::GmFree((void *)workspace);
//     AscendC::GmFree((void *)tiling);
// }

// workspace nullptr
TEST_F(moe_init_routing_v3_test, test_case_7)
{
    size_t num_rows = 1;
    size_t cols = 7168;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 2000000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system("chmod -R 755 ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 1 7168 8 256 float16"); // bf16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case6");

    std::string curPath = ".";
    ReadFile(curPath + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(curPath + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, 0, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, nullptr, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}
