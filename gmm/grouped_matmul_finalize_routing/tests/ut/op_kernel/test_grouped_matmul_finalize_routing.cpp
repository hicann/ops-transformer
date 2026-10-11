/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include <iostream>
#include <string>
#include "data_utils.h"
#include "test_grouped_matmul_finalize_routing_tiling_def.h"
#include "string.h"
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
                                                                      GM_ADDR pertoken_scale, GM_ADDR group_list,
                                                                      GM_ADDR share_input, GM_ADDR logit,
                                                                      GM_ADDR row_index, GM_ADDR offset, GM_ADDR y,
                                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM);
class grouped_matmul_finalize_routing_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_matmul_finalize_routing_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_matmul_finalize_routing_test TearDown\n" << std::endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(grouped_matmul_finalize_routing_test, test_grouped_matmul_finalize_routing_test1)
{
    system("cp -rf "
           "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/"
           "grouped_matmul_finalize_routing_data ./");

    system("chmod -R 755 ./grouped_matmul_finalize_routing_data/");
    system("cd ./grouped_matmul_finalize_routing_data/ && rm -rf ./*bin && python3 gen_data.py 1");

    char* path_ = get_current_dir_name();
    string path(path_);

    uint32_t m = 6144;
    uint32_t k = 2048;
    uint32_t n = 7168;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = m * k * sizeof(int8_t);
    size_t shape_w = 8 * k * n * sizeof(int8_t);
    size_t shape_scale = 8 * n * sizeof(float);
    size_t shape_pertoken_scale = m * 1 * sizeof(float);
    size_t shape_group_list = 8 * sizeof(int64_t);
    size_t shape_share_input = m * n * sizeof(uint16_t); // bf16
    size_t shape_logit = m * 1 * sizeof(float);
    size_t shape_row_index = m * 1 * sizeof(int64_t);

    size_t shape_y = m * n * sizeof(float);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 4 * 256 * 128 * sizeof(int32_t) * 24;
    size_t allWorkspaceSize = sysWorkspaceSize + usrWorkspaceSize;
    size_t tilingSize = sizeof(GroupMatmulFRTilingData);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(shape_w);
    uint8_t* scaleGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* pertoken_scaleGM = (uint8_t*)AscendC::GmAlloc(shape_pertoken_scale);
    uint8_t* group_listGM = (uint8_t*)AscendC::GmAlloc(shape_group_list);
    uint8_t* share_inputGM = (uint8_t*)AscendC::GmAlloc(shape_share_input);
    uint8_t* logitGM = (uint8_t*)AscendC::GmAlloc(shape_logit);
    uint8_t* row_indexGM = (uint8_t*)AscendC::GmAlloc(shape_row_index);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* contextGM = (uint8_t*)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(xGM, 0, shape_x);
    memset(wGM, 0, shape_w);
    memset(scaleGM, 0, shape_scale);
    memset(pertoken_scaleGM, 0, shape_pertoken_scale);
    memset(group_listGM, 0, shape_group_list);
    memset(share_inputGM, 0, shape_share_input);
    memset(logitGM, 0, shape_logit);
    memset(row_indexGM, 0, shape_row_index);
    memset(yGM, 0, shape_y);

    ReadFile(path + "/grouped_matmul_finalize_routing_data/x.bin", shape_x, xGM, shape_x);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/weight.bin", shape_w, wGM, shape_w);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/scale.bin", shape_scale, scaleGM, shape_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/groupList.bin", shape_group_list, pertoken_scaleGM,
             shape_group_list);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/perTokenScale.bin", shape_pertoken_scale, pertoken_scaleGM,
             shape_pertoken_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/logits.bin", shape_logit, logitGM, shape_logit);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/residual.bin", shape_share_input, share_inputGM,
             shape_share_input);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/sourceRow.bin", shape_row_index, row_indexGM,
             shape_row_index);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/y.bin", shape_y, yGM, shape_y);

    GroupMatmulFRTilingData* tilingData = reinterpret_cast<GroupMatmulFRTilingData*>(tiling);
    tilingData->coreNum = 24;
    tilingData->groupNum = 8;
    tilingData->totalInGroup = 576;
    tilingData->batch = 72;
    tilingData->k = 512;
    tilingData->n = 7168;
    tilingData->vBaseM = 16;
    tilingData->ubCalSize = 4096;
    tilingData->ubRestBytes = 126976;
    tilingData->parallNum = 4;
    tilingData->sharedInputOffset = 36;
    tilingData->sharedInputLen = 18;
    tilingData->residualScale = 1.0;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 128;
    tilingData->matmulTiling.N = 7168;
    tilingData->matmulTiling.Ka = 512;
    tilingData->matmulTiling.Kb = 512;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 7168;
    tilingData->matmulTiling.singleCoreK = 512;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 128;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 1;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 196608;
    tilingData->matmulTiling.shareL0CSize = 131072;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    ICPU_SET_TILING_KEY(10000000000000000001UL);
    ICPU_RUN_KF(grouped_matmul_finalize_routing, 24, xGM, wGM, scaleGM, nullptr, pertoken_scaleGM, group_listGM,
                share_inputGM, logitGM, row_indexGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)pertoken_scaleGM);
    AscendC::GmFree((void*)group_listGM);
    AscendC::GmFree((void*)share_inputGM);
    AscendC::GmFree((void*)logitGM);
    AscendC::GmFree((void*)row_indexGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}

TEST_F(grouped_matmul_finalize_routing_test, test_grouped_matmul_finalize_routing_test2)
{
    system("cp -rf "
           "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/"
           "grouped_matmul_finalize_routing_data ./");
    system("chmod -R 755 ./grouped_matmul_finalize_routing_data/");
    system("cd ./grouped_matmul_finalize_routing_data/ && rm -rf ./*bin && python3 gen_data.py 0");

    char* path_ = get_current_dir_name();
    string path(path_);

    uint32_t m = 1024;
    uint32_t k = 1024;
    uint32_t n = 8192;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = m * k * sizeof(int8_t);
    size_t shape_w = 1 * k * n * sizeof(int8_t);
    size_t shape_scale = 1 * n * sizeof(float);
    size_t shape_pertoken_scale = m * 1 * sizeof(float);
    size_t shape_group_list = 1 * sizeof(int64_t);
    size_t shape_row_index = m * 1 * sizeof(int64_t);

    size_t shape_y = m * n * sizeof(float);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 4 * 256 * 128 * sizeof(int32_t) * 24;
    size_t allWorkspaceSize = sysWorkspaceSize + usrWorkspaceSize;
    size_t tilingSize = sizeof(GroupMatmulFRTilingData);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(shape_w);
    uint8_t* scaleGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* pertoken_scaleGM = (uint8_t*)AscendC::GmAlloc(shape_pertoken_scale);
    uint8_t* group_listGM = (uint8_t*)AscendC::GmAlloc(shape_group_list);
    uint8_t* row_indexGM = (uint8_t*)AscendC::GmAlloc(shape_row_index);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* contextGM = (uint8_t*)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(xGM, 0, shape_x);
    memset(wGM, 0, shape_w);
    memset(scaleGM, 0, shape_scale);
    memset(pertoken_scaleGM, 0, shape_pertoken_scale);
    memset(group_listGM, 0, shape_group_list);
    memset(row_indexGM, 0, shape_row_index);
    memset(yGM, 0, shape_y);

    ReadFile(path + "/grouped_matmul_finalize_routing_data/x.bin", shape_x, xGM, shape_x);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/weight.bin", shape_w, wGM, shape_w);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/scale.bin", shape_scale, scaleGM, shape_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/groupList.bin", shape_group_list, group_listGM,
             shape_group_list);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/perTokenScale.bin", shape_pertoken_scale, pertoken_scaleGM,
             shape_pertoken_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/sourceRow.bin", shape_row_index, row_indexGM,
             shape_row_index);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/y.bin", shape_y, yGM, shape_y);

    GroupMatmulFRTilingData* tilingData = reinterpret_cast<GroupMatmulFRTilingData*>(tiling);
    tilingData->coreNum = 24;
    tilingData->groupNum = 1;
    tilingData->totalInGroup = 14024;
    tilingData->batch = 1024;
    tilingData->k = 1024;
    tilingData->n = 8192;
    tilingData->vBaseM = 16;
    tilingData->ubCalSize = 4096;
    tilingData->ubRestBytes = 126976;
    tilingData->parallNum = 4;
    tilingData->sharedInputOffset = 0;
    tilingData->sharedInputLen = 0;
    tilingData->residualScale = 1.0;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 128;
    tilingData->matmulTiling.N = 8192;
    tilingData->matmulTiling.Ka = 1024;
    tilingData->matmulTiling.Kb = 1024;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 8192;
    tilingData->matmulTiling.singleCoreK = 1024;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 128;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 1;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 262144;
    tilingData->matmulTiling.shareL0CSize = 131072;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    ICPU_SET_TILING_KEY(10000000000000000001UL);
    ICPU_RUN_KF(grouped_matmul_finalize_routing, 24, xGM, wGM, scaleGM, nullptr, pertoken_scaleGM, group_listGM,
                nullptr, nullptr, row_indexGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)pertoken_scaleGM);
    AscendC::GmFree((void*)group_listGM);
    AscendC::GmFree((void*)row_indexGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}

TEST_F(grouped_matmul_finalize_routing_test, test_grouped_matmul_finalize_routing_test3_int32_row_index)
{
    system("cp -rf "
           "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/"
           "grouped_matmul_finalize_routing_data ./");
    system("chmod -R 755 ./grouped_matmul_finalize_routing_data/");
    system("cd ./grouped_matmul_finalize_routing_data/ && rm -rf ./*bin && python3 gen_data.py 1");

    char* path_ = get_current_dir_name();
    string path(path_);

    uint32_t m = 6144;
    uint32_t k = 2048;
    uint32_t n = 7168;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = m * k * sizeof(int8_t);
    size_t shape_w = 8 * k * n * sizeof(int8_t);
    size_t shape_scale = 8 * n * sizeof(float);
    size_t shape_pertoken_scale = m * 1 * sizeof(float);
    size_t shape_group_list = 8 * sizeof(int64_t);
    size_t shape_share_input = m * n * sizeof(uint16_t); // bf16
    size_t shape_logit = m * 1 * sizeof(float);
    size_t shape_row_index = m * 1 * sizeof(int32_t);

    size_t shape_y = m * 7168 * sizeof(float);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 4 * 256 * 128 * sizeof(int32_t) * 24;
    size_t allWorkspaceSize = sysWorkspaceSize + usrWorkspaceSize;
    size_t tilingSize = sizeof(GroupMatmulFRTilingData);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(shape_w);
    uint8_t* scaleGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* pertoken_scaleGM = (uint8_t*)AscendC::GmAlloc(shape_pertoken_scale);
    uint8_t* group_listGM = (uint8_t*)AscendC::GmAlloc(shape_group_list);
    uint8_t* share_inputGM = (uint8_t*)AscendC::GmAlloc(shape_share_input);
    uint8_t* logitGM = (uint8_t*)AscendC::GmAlloc(shape_logit);
    uint8_t* row_indexGM = (uint8_t*)AscendC::GmAlloc(shape_row_index);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* contextGM = (uint8_t*)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(xGM, 0, shape_x);
    memset(wGM, 0, shape_w);
    memset(scaleGM, 0, shape_scale);
    memset(pertoken_scaleGM, 0, shape_pertoken_scale);
    memset(group_listGM, 0, shape_group_list);
    memset(share_inputGM, 0, shape_share_input);
    memset(logitGM, 0, shape_logit);
    memset(row_indexGM, 0, shape_row_index);
    memset(yGM, 0, shape_y);

    ReadFile(path + "/grouped_matmul_finalize_routing_data/x.bin", shape_x, xGM, shape_x);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/weight.bin", shape_w, wGM, shape_w);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/scale.bin", shape_scale, scaleGM, shape_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/groupList.bin", shape_group_list, group_listGM,
             shape_group_list);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/perTokenScale.bin", shape_pertoken_scale, pertoken_scaleGM,
             shape_pertoken_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/logits.bin", shape_logit, logitGM, shape_logit);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/residual.bin", shape_share_input, share_inputGM,
             shape_share_input);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/sourceRow_int32.bin", shape_row_index, row_indexGM,
             shape_row_index);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/y.bin", shape_y, yGM, shape_y);

    GroupMatmulFRTilingData* tilingData = reinterpret_cast<GroupMatmulFRTilingData*>(tiling);
    tilingData->coreNum = 24;
    tilingData->groupNum = 8;
    tilingData->totalInGroup = 576;
    tilingData->batch = 72;
    tilingData->k = 512;
    tilingData->n = 7168;
    tilingData->vBaseM = 16;
    tilingData->ubCalSize = 4096;
    tilingData->ubRestBytes = 126976;
    tilingData->parallNum = 4;
    tilingData->sharedInputOffset = 36;
    tilingData->sharedInputLen = 18;
    tilingData->residualScale = 1.0;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 128;
    tilingData->matmulTiling.N = 7168;
    tilingData->matmulTiling.Ka = 512;
    tilingData->matmulTiling.Kb = 512;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 7168;
    tilingData->matmulTiling.singleCoreK = 512;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 128;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 1;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 196608;
    tilingData->matmulTiling.shareL0CSize = 131072;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    ICPU_SET_TILING_KEY(10000000000000000011UL);
    ICPU_RUN_KF(grouped_matmul_finalize_routing, 24, xGM, wGM, scaleGM, nullptr, pertoken_scaleGM, group_listGM,
                share_inputGM, logitGM, row_indexGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)pertoken_scaleGM);
    AscendC::GmFree((void*)group_listGM);
    AscendC::GmFree((void*)share_inputGM);
    AscendC::GmFree((void*)logitGM);
    AscendC::GmFree((void*)row_indexGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}

TEST_F(grouped_matmul_finalize_routing_test, test_grouped_matmul_finalize_routing_test4_w4a8)
{
    system("cp -rf "
           "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/"
           "grouped_matmul_finalize_routing_data ./");
    system("chmod -R 755 ./grouped_matmul_finalize_routing_data/");
    system("cd ./grouped_matmul_finalize_routing_data/ && rm -rf ./*bin && python3 w4a8_gen_data.py");

    char* path_ = get_current_dir_name();
    string path(path_);

    uint32_t m = 576;
    uint32_t k = 2048;
    uint32_t n = 7168;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = m * k * sizeof(int8_t);
    size_t shape_w = 8 * k * n * sizeof(int8_t) / 2;
    size_t shape_scale = 8 * n * sizeof(int64_t);
    size_t shape_bias = 8 * n * sizeof(float);
    size_t shape_pertoken_scale = m * 1 * sizeof(float);
    size_t shape_group_list = 8 * sizeof(int64_t);
    size_t shape_share_input = m * n * sizeof(uint16_t); // bf16
    size_t shape_logit = m * 1 * sizeof(float);
    size_t shape_row_index = m * 1 * sizeof(int64_t);

    size_t shape_y = m * n * sizeof(float);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 4 * 256 * 128 * sizeof(int32_t) * 24;
    size_t allWorkspaceSize = sysWorkspaceSize + usrWorkspaceSize;
    size_t tilingSize = sizeof(GroupMatmulFRTilingData);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(shape_w);
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(shape_bias);
    uint8_t* scaleGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* pertoken_scaleGM = (uint8_t*)AscendC::GmAlloc(shape_pertoken_scale);
    uint8_t* group_listGM = (uint8_t*)AscendC::GmAlloc(shape_group_list);
    uint8_t* share_inputGM = (uint8_t*)AscendC::GmAlloc(shape_share_input);
    uint8_t* logitGM = (uint8_t*)AscendC::GmAlloc(shape_logit);
    uint8_t* row_indexGM = (uint8_t*)AscendC::GmAlloc(shape_row_index);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* contextGM = (uint8_t*)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(xGM, 0, shape_x);
    memset(wGM, 0, shape_w);
    memset(biasGM, 0, shape_bias);
    memset(scaleGM, 0, shape_scale);
    memset(pertoken_scaleGM, 0, shape_pertoken_scale);
    memset(group_listGM, 0, shape_group_list);
    memset(share_inputGM, 0, shape_share_input);
    memset(logitGM, 0, shape_logit);
    memset(row_indexGM, 0, shape_row_index);
    memset(yGM, 0, shape_y);

    ReadFile(path + "/grouped_matmul_finalize_routing_data/x.bin", shape_x, xGM, shape_x);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/weight.bin", shape_w, wGM, shape_w);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/scale.bin", shape_scale, scaleGM, shape_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/bias.bin", shape_bias, biasGM, shape_bias);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/groupList.bin", shape_group_list, group_listGM,
             shape_group_list);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/perTokenScale.bin", shape_pertoken_scale, pertoken_scaleGM,
             shape_pertoken_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/logits.bin", shape_logit, logitGM, shape_logit);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/residual.bin", shape_share_input, share_inputGM,
             shape_share_input);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/sourceRow_int64.bin", shape_row_index, row_indexGM,
             shape_row_index);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/y.bin", shape_y, yGM, shape_y);

    GroupMatmulFRTilingData* tilingData = reinterpret_cast<GroupMatmulFRTilingData*>(tiling);
    tilingData->coreNum = 24;
    tilingData->groupNum = 8;
    tilingData->totalInGroup = 576;
    tilingData->batch = 72;
    tilingData->k = 2048;
    tilingData->n = 7168;
    tilingData->vBaseM = 16;
    tilingData->ubCalSize = 4096;
    tilingData->ubRestBytes = 126976;
    tilingData->parallNum = 4;
    tilingData->sharedInputOffset = 36;
    tilingData->sharedInputLen = 18;
    tilingData->residualScale = 1.0;
    tilingData->quantGroupNum = 256;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 576;
    tilingData->matmulTiling.N = 7168;
    tilingData->matmulTiling.Ka = 512;
    tilingData->matmulTiling.Kb = 512;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 7168;
    tilingData->matmulTiling.singleCoreK = 512;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 128;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 1;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 196608;
    tilingData->matmulTiling.shareL0CSize = 131072;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    ICPU_SET_TILING_KEY(11000000000000000011UL);
    ICPU_RUN_KF(grouped_matmul_finalize_routing, 24, xGM, wGM, scaleGM, biasGM, pertoken_scaleGM, group_listGM,
                share_inputGM, logitGM, row_indexGM, nullptr, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)pertoken_scaleGM);
    AscendC::GmFree((void*)group_listGM);
    AscendC::GmFree((void*)share_inputGM);
    AscendC::GmFree((void*)logitGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)row_indexGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}

TEST_F(grouped_matmul_finalize_routing_test, test_grouped_matmul_finalize_routing_test5_w4a8_offset)
{
    system("cp -rf "
           "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/"
           "grouped_matmul_finalize_routing_data ./");
    system("chmod -R 755 ./grouped_matmul_finalize_routing_data/");
    system("cd ./grouped_matmul_finalize_routing_data/ && rm -rf ./*bin && python3 w4a8_gen_data.py");

    char* path_ = get_current_dir_name();
    string path(path_);

    uint32_t m = 576;
    uint32_t k = 2048;
    uint32_t n = 7168;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = m * k * sizeof(int8_t);
    size_t shape_w = 8 * k * n * sizeof(int8_t) / 2;
    size_t shape_scale = 8 * n * sizeof(int64_t);
    size_t shape_offset = 8 * n * sizeof(float);
    size_t shape_bias = 8 * n * sizeof(float);
    size_t shape_pertoken_scale = m * 1 * sizeof(float);
    size_t shape_group_list = 8 * sizeof(int64_t);
    size_t shape_share_input = m * n * sizeof(uint16_t); // bf16
    size_t shape_logit = m * 1 * sizeof(float);
    size_t shape_row_index = m * 1 * sizeof(int64_t);

    size_t shape_y = m * 7168 * sizeof(float);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 4 * 256 * 128 * sizeof(int32_t) * 24 * 2 * 8 + 576 * sizeof(float);
    size_t allWorkspaceSize = sysWorkspaceSize + usrWorkspaceSize;
    size_t tilingSize = sizeof(GroupMatmulFRTilingData);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(shape_w);
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(shape_bias);
    uint8_t* scaleGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* offsetGM = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* pertoken_scaleGM = (uint8_t*)AscendC::GmAlloc(shape_pertoken_scale);
    uint8_t* group_listGM = (uint8_t*)AscendC::GmAlloc(shape_group_list);
    uint8_t* share_inputGM = (uint8_t*)AscendC::GmAlloc(shape_share_input);
    uint8_t* logitGM = (uint8_t*)AscendC::GmAlloc(shape_logit);
    uint8_t* row_indexGM = (uint8_t*)AscendC::GmAlloc(shape_row_index);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* contextGM = (uint8_t*)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(xGM, 0, shape_x);
    memset(wGM, 0, shape_w);
    memset(biasGM, 0, shape_bias);
    memset(scaleGM, 0, shape_scale);
    memset(offsetGM, 0, shape_offset);
    memset(pertoken_scaleGM, 0, shape_pertoken_scale);
    memset(group_listGM, 0, shape_group_list);
    memset(share_inputGM, 0, shape_share_input);
    memset(logitGM, 0, shape_logit);
    memset(row_indexGM, 0, shape_row_index);
    memset(yGM, 0, shape_y);

    ReadFile(path + "/grouped_matmul_finalize_routing_data/x.bin", shape_x, xGM, shape_x);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/weight.bin", shape_w, wGM, shape_w);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/scale.bin", shape_scale, scaleGM, shape_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/bias.bin", shape_bias, biasGM, shape_bias);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/offset.bin", shape_offset, offsetGM, shape_offset);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/groupList.bin", shape_group_list, group_listGM,
             shape_group_list);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/perTokenScale.bin", shape_pertoken_scale, pertoken_scaleGM,
             shape_pertoken_scale);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/logits.bin", shape_logit, logitGM, shape_logit);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/residual.bin", shape_share_input, share_inputGM,
             shape_share_input);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/sourceRow_int64.bin", shape_row_index, row_indexGM,
             shape_row_index);
    ReadFile(path + "/grouped_matmul_finalize_routing_data/y.bin", shape_y, yGM, shape_y);

    GroupMatmulFRTilingData* tilingData = reinterpret_cast<GroupMatmulFRTilingData*>(tiling);
    tilingData->coreNum = 24;
    tilingData->groupNum = 8;
    tilingData->totalInGroup = 576;
    tilingData->batch = 72;
    tilingData->k = 2048;
    tilingData->n = 7168;
    tilingData->vBaseM = 16;
    tilingData->ubCalSize = 4096;
    tilingData->ubRestBytes = 126976;
    tilingData->parallNum = 4;
    tilingData->sharedInputOffset = 36;
    tilingData->sharedInputLen = 18;
    tilingData->residualScale = 1.0;
    tilingData->quantGroupNum = 256;
    tilingData->withOffset = 1;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 576;
    tilingData->matmulTiling.N = 7168;
    tilingData->matmulTiling.Ka = 512;
    tilingData->matmulTiling.Kb = 512;
    tilingData->matmulTiling.singleCoreM = 128;
    tilingData->matmulTiling.singleCoreN = 7168;
    tilingData->matmulTiling.singleCoreK = 512;
    tilingData->matmulTiling.baseM = 128;
    tilingData->matmulTiling.baseN = 256;
    tilingData->matmulTiling.baseK = 128;
    tilingData->matmulTiling.depthA1 = 8;
    tilingData->matmulTiling.depthB1 = 8;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 1;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 196608;
    tilingData->matmulTiling.shareL0CSize = 131072;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 4;
    tilingData->matmulTiling.stepKb = 4;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    ICPU_SET_TILING_KEY(11000000000000000011UL);
    ICPU_RUN_KF(grouped_matmul_finalize_routing, 24, xGM, wGM, scaleGM, biasGM, pertoken_scaleGM, group_listGM,
                share_inputGM, logitGM, row_indexGM, offsetGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)offsetGM);
    AscendC::GmFree((void*)pertoken_scaleGM);
    AscendC::GmFree((void*)group_listGM);
    AscendC::GmFree((void*)share_inputGM);
    AscendC::GmFree((void*)logitGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)row_indexGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}
