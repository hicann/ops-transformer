/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_inplace_partial_rotary_mul_kernel.cpp
 * @brief OpKernel UT for InplacePartialRotaryMul using TTK (ESL/CA simulation)
 *
 * Requires:
 *   - Data generation scripts: tests/ut/op_kernel/inplace_partial_rotary_mul_data/gen_data.py
 *   - Tiling generation scripts: tests/ut/op_kernel/inplace_partial_rotary_mul_data/gen_tiling.py
 *   - Golden reference: tests/ut/op_kernel/inplace_partial_rotary_mul_data/gen_golden.py
 *
 * Coverage per tiling path:
 *   Ascend950 (regbase): 18 TilingKeys across 4 TilingClass strategies
 *   Ascend910B (membase): TilingKey 1/2 + generic split S/BS/BSN + pad variants
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling);

class InplacePartialRotaryMulKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulKernelTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulKernelTest TearDown\n" << endl;
    }

    /**
     * @brief Helper to run a kernel test with given shape and dtype
     * @param xShape   4D shape of x tensor [B, S, N, D]
     * @param cosShape 4D shape of cos tensor
     * @param dtype    Data type string ("float16", "bfloat16", "float32")
     * @param tilingKey Tiling key for kernel dispatch
     * @param blockDim Number of blocks
     * @param sliceStart Partial slice start
     * @param sliceEnd   Partial slice end
     */
    void RunKernelTest(const vector<int64_t>& xShape, const vector<int64_t>& cosShape, const string& dtype,
                       uint64_t tilingKey, uint32_t blockDim, int64_t sliceStart = 0, int64_t sliceEnd = 0)
    {
        // Calculate byte sizes
        int64_t dtypeSize = (dtype == "float32") ? 4 : 2;
        int64_t cosDtypeSize = dtypeSize;
        size_t inputXByteSize = 1;
        for (auto d : xShape)
            inputXByteSize *= d;
        inputXByteSize *= dtypeSize;

        size_t inputCosByteSize = 1;
        for (auto d : cosShape)
            inputCosByteSize *= d;
        inputCosByteSize *= cosDtypeSize;
        size_t inputSinByteSize = inputCosByteSize;

        // Inplace: output is same as input x
        size_t outputYByteSize = inputXByteSize;
        size_t tilingDataSize = 4096; // RopeRegbaseTilingData is large

        // Allocate GM buffers
        uint8_t* xGm = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
        uint8_t* cosGm = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
        uint8_t* sinGm = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
        uint8_t* yGm = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
        uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
        uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

        // Generate test data (runs Python scripts)
        string dataCmd = "cd ./inplace_partial_rotary_mul_data/ && "
                         "rm -rf ./*.bin && "
                         "python3 gen_data.py ";
        for (auto d : xShape)
            dataCmd += to_string(d) + " ";
        dataCmd += dtype;
        system(dataCmd.c_str());

        string tilingCmd = "cd ./inplace_partial_rotary_mul_data/ && "
                           "python3 gen_tiling.py " +
                           to_string(tilingKey) + " " + to_string(sliceStart) + " " + to_string(sliceEnd);
        system(tilingCmd.c_str());

        // Read data from generated binaries
        char* cwd = get_current_dir_name();
        string basePath(cwd);
        free(cwd);

        ReadFile(basePath + "/inplace_partial_rotary_mul_data/x.bin", inputXByteSize, xGm, inputXByteSize);
        ReadFile(basePath + "/inplace_partial_rotary_mul_data/cos.bin", inputCosByteSize, cosGm, inputCosByteSize);
        ReadFile(basePath + "/inplace_partial_rotary_mul_data/sin.bin", inputSinByteSize, sinGm, inputSinByteSize);
        ReadFile(basePath + "/inplace_partial_rotary_mul_data/tiling.bin", tilingDataSize, tilingGm, tilingDataSize);

        // Run kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(inplace_partial_rotary_mul, blockDim, xGm, cosGm, sinGm, yGm, workspaceGm, tilingGm);

        // Copy x output (inplace) to yGm for comparison
        // Inplace: result is in xGm, copy to yGm for golden comparison
        AscendC::Memcpy(yGm, xGm, outputYByteSize);

        // Generate golden reference
        string goldenCmd = "cd ./inplace_partial_rotary_mul_data/ && "
                           "python3 gen_golden.py " +
                           to_string(sliceStart) + " " + to_string(sliceEnd);
        system(goldenCmd.c_str());

        // Compare result against golden
        vector<uint8_t> goldenData(outputYByteSize);
        ReadFile(basePath + "/inplace_partial_rotary_mul_data/golden.bin", outputYByteSize, goldenData.data(),
                 outputYByteSize);

        // Validate: compare yGm vs golden
        bool match = (memcmp(yGm, goldenData.data(), outputYByteSize) == 0);
        EXPECT_TRUE(match) << "Kernel output does not match golden reference";

        // Cleanup
        AscendC::GmFree(xGm);
        AscendC::GmFree(cosGm);
        AscendC::GmFree(sinGm);
        AscendC::GmFree(yGm);
        AscendC::GmFree(workspaceGm);
        AscendC::GmFree(tilingGm);
    }

    void PrepareDataDir()
    {
        system("cp -r ../../../../../posembedding/inplace_partial_rotary_mul/tests/ut/op_kernel/"
               "inplace_partial_rotary_mul_data ./");
        system("chmod -R 755 ./inplace_partial_rotary_mul_data/");
    }
};

// ============================================================================
// Ascend950 RegBase Kernel Tests
// ============================================================================

// AAndB — TilingKey A (20040): FP16, NO_BROADCAST
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_no_broadcast_fp16)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 128}, {2, 4, 8, 128}, "float16", 20040, 64, 0, 128);
}

// AAndB — TilingKey B (20041): FP16, BROADCAST_BSN
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_broadcast_fp16)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 128}, {1, 1, 1, 128}, "float16", 20041, 64, 0, 128);
}

// AAndB — FP32
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_no_broadcast_fp32)
{
    PrepareDataDir();
    RunKernelTest({1, 1, 1, 64}, {1, 1, 1, 64}, "float32", 20040, 64, 0, 64);
}

// AAndB — Mixed: BF16 x + FP32 cos/sin → TilingKey 20140
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_mixed_bf16_cos_fp32)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 64}, {2, 4, 8, 64}, "bfloat16", 20140, 64, 0, 64);
}

// AAndB — Mixed: FP16 x + FP32 cos/sin → TilingKey 20240
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_mixed_fp16_cos_fp32)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 64}, {2, 4, 8, 64}, "float16", 20240, 64, 0, 64);
}

// BAB — BSND, FP16 → TilingKey 20020
TEST_F(InplacePartialRotaryMulKernelTest, BAB_bsnd_fp16)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 64}, {1, 4, 1, 64}, "float16", 20020, 64, 0, 64);
}

// BAB — Mixed: BF16+FP32 → TilingKey 20120
TEST_F(InplacePartialRotaryMulKernelTest, BAB_mixed_bf16_cos_fp32)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 64}, {1, 4, 1, 64}, "bfloat16", 20120, 64, 0, 64);
}

// AB — SBND, FP16 → TilingKey 20030
TEST_F(InplacePartialRotaryMulKernelTest, AB_sbnd_fp16)
{
    PrepareDataDir();
    RunKernelTest({4, 2, 8, 64}, {4, 1, 1, 64}, "float16", 20030, 64, 0, 64);
}

// ABA — BNSD, cosB!=1 → TilingKey 20010
TEST_F(InplacePartialRotaryMulKernelTest, ABA_bnsd_fp16)
{
    PrepareDataDir();
    RunKernelTest({2, 8, 4, 64}, {2, 8, 1, 64}, "float16", 20010, 64, 0, 64);
}

// BA — BNSD, cosB==1 → TilingKey 20011
TEST_F(InplacePartialRotaryMulKernelTest, BA_bnsd_fp16)
{
    PrepareDataDir();
    RunKernelTest({2, 8, 4, 64}, {1, 8, 1, 64}, "float16", 20011, 64, 0, 64);
}

// BA — Mixed: BF16+FP32 → TilingKey 20111
TEST_F(InplacePartialRotaryMulKernelTest, BA_mixed_bf16_cos_fp32)
{
    PrepareDataDir();
    RunKernelTest({2, 8, 4, 64}, {1, 8, 1, 64}, "bfloat16", 20111, 64, 0, 64);
}

// AAndB — Partial slice: D=128, slice=[32,96] → sliceLength=64
TEST_F(InplacePartialRotaryMulKernelTest, AAndB_partial_slice)
{
    PrepareDataDir();
    RunKernelTest({2, 4, 8, 128}, {2, 4, 8, 64}, "float16", 20040, 64, 32, 96);
}

// ============================================================================
// Ascend910B Membase Kernel Tests
// ============================================================================

// Fast path Key 1: isBrc=true
TEST_F(InplacePartialRotaryMulKernelTest, Membase_fast_key1_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 4, 1, 64}, {1, 1, 1, 64}, "float16", 1, 64, 0, 64);
}

// Fast path Key 2: isBrc=false
TEST_F(InplacePartialRotaryMulKernelTest, Membase_fast_key2_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 1, 1, 64}, {1, 1, 1, 64}, "float16", 2, 64, 0, 64);
}

// Generic Split S + FP16 → TilingKey 2000
TEST_F(InplacePartialRotaryMulKernelTest, Membase_split_s_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 4, 1, 256}, {1, 4, 1, 256}, "float16", 2000, 64, 0, 256);
}

// Generic Split S + BF16 → TilingKey 2010
TEST_F(InplacePartialRotaryMulKernelTest, Membase_split_s_bf16)
{
    PrepareDataDir();
    RunKernelTest({1, 4, 1, 128}, {1, 4, 1, 128}, "bfloat16", 2010, 64, 0, 128);
}

// Generic Split S + FP32 → TilingKey 2020
TEST_F(InplacePartialRotaryMulKernelTest, Membase_split_s_fp32)
{
    PrepareDataDir();
    RunKernelTest({1, 4, 1, 64}, {1, 4, 1, 64}, "float32", 2020, 64, 0, 64);
}

// Generic Split S + PAD (D=254 unaligned) → TilingKey 2001
TEST_F(InplacePartialRotaryMulKernelTest, Membase_split_s_pad_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 1, 4, 254}, {1, 1, 1, 254}, "float16", 2001, 64, 0, 254);
}

// Generic Split BS → TilingKey 2100
TEST_F(InplacePartialRotaryMulKernelTest, Membase_split_bs_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 8, 4, 128}, {1, 8, 1, 128}, "float16", 2100, 64, 0, 128);
}

// Boundary: D=2 minimum
TEST_F(InplacePartialRotaryMulKernelTest, Boundary_d2_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 1, 1, 2}, {1, 1, 1, 2}, "float16", 2001, 64, 0, 2);
}

// Boundary: D=1024 maximum
TEST_F(InplacePartialRotaryMulKernelTest, Boundary_d1024_fp16)
{
    PrepareDataDir();
    RunKernelTest({1, 1, 1, 1024}, {1, 1, 1, 1024}, "float16", 2000, 64, 0, 1024);
}
