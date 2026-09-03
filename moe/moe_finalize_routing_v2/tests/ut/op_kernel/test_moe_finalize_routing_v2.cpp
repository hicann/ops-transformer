/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "moe_finalize_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include <cstdint>
#include "tikicpulib.h"
#include "data_utils.h"
#endif

extern "C" __global__ __aicore__ void moe_finalize_routing_v2(GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow,
                                                              GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                                              GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out,
                                                              GM_ADDR workspace, GM_ADDR tiling);

class moe_finalize_routing_test_v2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "moe_finalize_routing_test_v2 SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "moe_finalize_routing_test_v2 TearDown\n" << std::endl;
    }
};

TEST_F(moe_finalize_routing_test_v2, test_case_float32_db1)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case0'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20004);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_db1)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case1'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20005);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_db1)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case2'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20003);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth_k2)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case3'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20006);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth_k2)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case4'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20007);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth_k2)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case5'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20008);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth_k4)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case6'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20009);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth_k4)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case7'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20010);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth_k4)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case8'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20011);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case9'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20012);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case10'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20013);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case11'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20014);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cutk)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case12'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20000);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cutk)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case13'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20001);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cutk)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case14'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20002);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_all_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 471 30 4 933 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case15'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    uint32_t blockDim = 39;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20015);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_all_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 702 30 5 340 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 340;
    uint32_t H = 30;
    uint32_t K = 5;
    uint32_t E = 702;
    uint32_t blockDim = 38;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20016);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_all_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 592 30 6 105 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case17'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    uint32_t blockDim = 35;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20017);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_networkCase)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 16 20 4 12 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case18'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12;
    uint32_t H = 20;
    uint32_t K = 4;
    uint32_t E = 16;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20018);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_db1_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case19'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20023);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_db1_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case20'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20024);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_db1_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case21'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20022);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth_k2_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case22'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20025);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth_k2_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case23'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20026);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth_k2_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case24'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20027);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth_k4_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case25'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20028);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth_k4_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case26'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20029);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth_k4_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case27'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20030);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cuth_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 1024 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case28'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20031);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cuth_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 1024 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case29'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 1024; // 2;
    uint32_t H = 30;    // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 40;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20032);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cuth_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 10 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case30'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20033);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_cutk_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case31'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20019);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_cutk_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20020);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_cutk_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case33'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20021);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float32_all_bias_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 471 30 4 933 'float32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case34'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    uint32_t blockDim = 39;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(float);
    size_t skip1ByteSize = BS * H * sizeof(float);
    size_t skip2ByteSize = BS * H * sizeof(float);
    size_t biasByteSize = E * H * sizeof(float);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(float);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20034);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_float16_all_bias_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 702 30 5 340 'float16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case35'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 340;
    uint32_t H = 30;
    uint32_t K = 5;
    uint32_t E = 702;
    uint32_t blockDim = 38;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20035);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_bfloat16_all_bias_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 592 30 6 105 'bfloat16' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case36'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    uint32_t blockDim = 35;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(half);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20036);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0037_bf16_fp32_cutk)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case37'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20002);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0038_bf16_fp32_db1)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case38'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20003);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0039_bf16_fp32_cuth_k2)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case39'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20008);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0040_bf16_fp32_cuth_k4)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case40'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20011);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0041_bf16_fp32_cuth)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case41'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20014);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0042_bf16_fp32_all_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 592 30 6 105 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case42'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    uint32_t blockDim = 35;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20017);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0043_bf16_fp32_cutk_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  300 7 257 12 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case43'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20021);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0044_bf16_fp32_db1_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  3 7 2 12 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case44'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 12; // 2;
    uint32_t H = 7;   // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20022);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0045_bf16_fp32_cuth_k2_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 2 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case45'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 2;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20027);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0046_bf16_fp32_cuth_k4_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 4 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case46'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 4;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20030);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0047_bf16_fp32_cuth_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py  10 30 3 10 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case47'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 10; // 2;
    uint32_t H = 30;  // 128 * 32;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 10;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20033);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(moe_finalize_routing_test_v2, test_case_0048_bf16_fp32_all_bias_without_bias)
{
    system("cp -rf "
           "../../../../../moe/moe_finalize_routing_v2/tests/ut/op_kernel/finalize_routing_v2_data ./");
    system("chmod -R 755 ./finalize_routing_v2_data/");
    system("cd ./finalize_routing_v2_data/ && python3 gen_data.py 592 30 6 105 'bf16fp32' 'true'");
    system("cd ./finalize_routing_v2_data/ && python3 gen_tiling.py 'case48'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeFinalizeRoutingV2TilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    uint32_t blockDim = 35;

    size_t expandedPermutedRowsByteSize = BS * K * H * sizeof(half);
    size_t skip1ByteSize = BS * H * sizeof(half);
    size_t skip2ByteSize = BS * H * sizeof(half);
    size_t biasByteSize = E * H * sizeof(half);
    size_t scalesByteSize = BS * K * sizeof(float);
    size_t expandedSrcToDstRowByteSize = BS * K * sizeof(int32_t);
    size_t expertForSourceRowByteSize = BS * K * sizeof(int32_t);
    size_t outByteSize = BS * H * sizeof(half);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);

    uint8_t *in_expanded_permuted_rows = (uint8_t *)AscendC::GmAlloc(expandedPermutedRowsByteSize);
    uint8_t *in_skip1 = (uint8_t *)AscendC::GmAlloc(skip1ByteSize);
    uint8_t *in_skip2 = (uint8_t *)AscendC::GmAlloc(skip2ByteSize);
    uint8_t *in_bias = (uint8_t *)AscendC::GmAlloc(biasByteSize);
    uint8_t *in_scales = (uint8_t *)AscendC::GmAlloc(scalesByteSize);
    uint8_t *in_expanded_src_to_dst_row = (uint8_t *)AscendC::GmAlloc(expandedSrcToDstRowByteSize);

    uint8_t *in_expert_for_source_row = (uint8_t *)AscendC::GmAlloc(expertForSourceRowByteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outByteSize);

    uint8_t *workSpace = (uint8_t *)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_permuted_rows.bin", expandedPermutedRowsByteSize,
             in_expanded_permuted_rows, expandedPermutedRowsByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip1.bin", skip1ByteSize, in_skip1, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/skip2_optional.bin", skip1ByteSize, in_skip2, skip1ByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/bias.bin", biasByteSize, in_bias, biasByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/scales.bin", scalesByteSize, in_scales, scalesByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expanded_src_to_dst_row.bin", expandedSrcToDstRowByteSize,
             in_expanded_src_to_dst_row, expandedSrcToDstRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/expert_for_source_row.bin", expertForSourceRowByteSize,
             in_expert_for_source_row, expertForSourceRowByteSize);
    ReadFile(curPath + "/finalize_routing_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(20036);
    ICPU_RUN_KF(moe_finalize_routing_v2, blockDim, in_expanded_permuted_rows, in_expanded_src_to_dst_row, in_skip1,
                in_skip2, in_bias, in_scales, in_expert_for_source_row, out, workSpace, tiling);

    WriteFile(curPath + "/finalize_routing_v2_data/output.bin", out, outByteSize);
    AscendC::GmFree((void *)in_expanded_permuted_rows);
    AscendC::GmFree((void *)in_skip1);
    AscendC::GmFree((void *)in_skip2);
    AscendC::GmFree((void *)in_bias);
    AscendC::GmFree((void *)in_scales);
    AscendC::GmFree((void *)in_expanded_src_to_dst_row);
    AscendC::GmFree((void *)in_expert_for_source_row);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workSpace);
    AscendC::GmFree((void *)tiling);
}
