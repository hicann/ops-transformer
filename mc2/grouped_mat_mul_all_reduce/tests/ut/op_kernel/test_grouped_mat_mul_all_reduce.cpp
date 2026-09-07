/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_mat_mul_all_reduce.h
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <string>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "grouped_mat_mul_all_reduce_tiling_def.h"
#include "grouped_matmul_tensorlist.h"
using namespace std;

extern "C" __global__ __aicore__ void grouped_mat_mul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                 GM_ADDR groupListGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                                 GM_ADDR tilingGM);

extern uint8_t *g_hcclContextReserved[2];

std::ostream &Print(const std::vector<uint64_t> &arr, char *name, std::ostream &out)
{
    out << std::string(name) << ": ";
    for (auto &i : arr) {
        out << i << " ";
    }
    out << "\n";
    return out;
}

template <class T>
std::ostream &Print(T arr[], char *name, uint32_t count, std::ostream &out)
{
    out << std::string(name) << ": ";
    for (uint32_t i = 0; i < count; ++i) {
        out << arr[i] << " ";
    }
    out << "\n";
    return out;
}

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

class GroupedMatMulAllReduceTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t *)AscendC::GmAlloc(ctxSize);
        cout << "GroupedMatMulAllReduceTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        AscendC::GmFree((void *)g_hcclContextReserved[0]);
        cout << "GroupedMatMulAllReduceTest TearDown\n" << endl;
    }
};

TEST_F(GroupedMatMulAllReduceTest, CaseFloat16Test1)
{
    std::vector<uint64_t> mList = {256, 1024};
    std::vector<uint64_t> kList = {256, 256};
    std::vector<uint64_t> nList = {256, 1024};

    std::vector<std::vector<uint64_t>> xShapeInfo;
    std::vector<std::vector<uint64_t>> weightShapeInfo;
    std::vector<std::vector<uint64_t>> yShapeInfo;
    for (int idx = 0; idx < mList.size(); ++idx) {
        xShapeInfo.push_back({mList[idx], kList[idx]});
        weightShapeInfo.push_back({kList[idx], nList[idx]});
        yShapeInfo.push_back({mList[idx], nList[idx]});
    }

    system("cp -r ../../../../../mc2/grouped_mat_mul_all_reduce/tests/ut/op_kernel/grouped_mat_mul_all_reduce_data ./");
    system("chmod -R 755 ./grouped_mat_mul_all_reduce_data/");
    system("rm -f ./grouped_mat_mul_all_reduce_data/*.bin");
    system("cd ./grouped_mat_mul_all_reduce_data/ && python3 gen_data.py '{256, 1024}' '{256, 256}' '{256, 1024}' "
           "'float16'");
    system("cd ./grouped_mat_mul_all_reduce_data/ && python3 gen_tiling.py case_float16_1");
    cout << "=================XXXXXXXXXXXXXXXXXXXXXXXX\n" << endl;
    system("ls -lh ./grouped_mat_mul_all_reduce_data/");
    cout << "=================XXXXXXXXXXXXXXXXXXXXXXXX\n" << endl;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t numBlocks = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(allWorkspaceSize));
    size_t tilingSize = sizeof(GMMAllReduceTilingData);
    uint8_t *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingSize));

    GMMAllReduceTilingData *tilingDataFromBin = reinterpret_cast<GMMAllReduceTilingData *>(tiling);
    ReadFile("./grouped_mat_mul_all_reduce_data/tiling.bin", tilingSize, tilingDataFromBin, tilingSize);
    uint32_t groupNum = tilingDataFromBin->aicoreTiling.baseParams.groupNum;
    std::cout << "coreNum = " << tilingDataFromBin->aicoreTiling.baseParams.coreNum << ", groupNum = " << groupNum
              << ", tilingSize = " << tilingSize << std::endl;
    Print(tilingDataFromBin->aicoreTiling.baseParams.mList, "mList", 10, std::cout);

    AllReduceMsg *msg = reinterpret_cast<AllReduceMsg *>(tilingDataFromBin->aicpuTiling.msg);
    uint32_t baseM = tilingDataFromBin->aicoreTiling.mmTilingData.baseM;

    for (uint32_t i = 0; i < groupNum; ++i) {
        AllReduceMsg *msgI = msg + i;
        msgI->debugMode = 4; // ASCEND_MC2_DEBUG_MODE
        msgI->commType = 2;  // HCCL_CMD_ALLREDUCE
        msgI->reduceOp = 0;  // HCCL_REDUCE_SUM
        msgI->waitPolicy = 1;
        msgI->rspPolicy = 1;
        msgI->exitPolicy = 0;
        msgI->commAlg = 0;
        msgI->taskType = 4;  // KFC_TASK_HCC_TASK_DELIVER
        msgI->commOrder = 1; // 0先AiCPU后MM;  1为先MM后AICPU
        msgI->notifyOff = 1;
        msgI->notifyBeginCnt = 1; // NOTIFY_WRITE_CNT
        msgI->notifyEndCnt = 1;
        msgI->funID = 2;    // ALL_REDUCE_FUNC_ID
        msgI->dataType = 3; // HCCL_DATA_TYPE_FP16
        msgI->groupNum = 2;
        msgI->stride = 0;
        msgI->useBufferType = 1; // MC2_BUFFER_TYPE_OUTPUT
        msgI->workspaceOff = 0;

        msgI->sendOff = baseM * nList[i] * sizeof(half);
        msgI->recvOff = baseM * nList[i] * sizeof(half);
        msgI->sendCnt = baseM * nList[i];
        msgI->recvCnt = baseM * nList[i];
        msgI->tailSendOff = 0;
        msgI->tailRecvOff = 0;
        msgI->tailSendCnt = 0;
        msgI->tailRecvCnt = 0;
        msgI->totalCnt = mList[i] * nList[i];
        if (i == 0) {
            msgI->reuseMode = 2;
            msgI->turnNum = 2;
            msgI->tailNum = 0;
        } else {
            msgI->reuseMode = 8;
            msgI->turnNum = 8;
            msgI->tailNum = 0;
        }
    }

    std::string baseDir("./grouped_mat_mul_all_reduce_data/");
    uint8_t *x = GROUPED_MATMUL::CreateTensorList<half>(xShapeInfo, "float16", "x", baseDir);
    uint8_t *weight = GROUPED_MATMUL::CreateTensorList<half>(weightShapeInfo, "float16", "weight", baseDir);
    uint8_t *bias = GROUPED_MATMUL::CreateTensorList<half>(nList, "float16", "bias", baseDir);
    // if not nullptr, use GROUPED_MATMUL::CreateTensorList<half>({}, "int64", "groupList", baseDir);
    uint8_t *groupList = nullptr;
    uint8_t *y = GROUPED_MATMUL::CreateTensorList<half>(yShapeInfo, "float16", "y", baseDir);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(grouped_mat_mul_all_reduce, numBlocks, x, weight, bias, groupList, y, workspace, tiling);

    GROUPED_MATMUL::FreeTensorList<half>(y, yShapeInfo, "float16", "y", baseDir);
    GROUPED_MATMUL::FreeTensorList<half>(x, xShapeInfo, "float16", "x", baseDir);
    GROUPED_MATMUL::FreeTensorList<half>(weight, weightShapeInfo, "float16", "weight", baseDir);
    GROUPED_MATMUL::FreeTensorList<half>(bias, nList, "float16", "bias", baseDir);
    AscendC::GmFree(reinterpret_cast<void *>(tiling));
    AscendC::GmFree(reinterpret_cast<void *>(workspace));

    system("cd ./grouped_mat_mul_all_reduce_data/ && python3 compare_data.py 'float16'");
}
