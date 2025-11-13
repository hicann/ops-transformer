/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_moe_distribute_buffer_reset.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <unordered_set>
#include <bits/stdc++.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_buffer_reset.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int DIE_PER_SERVER = 16;
constexpr uint32_t SERVER_NUM = 1;
constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t EP_WORLD_SIZE = WORLD_SIZE * SERVER_NUM;
constexpr uint32_t TP_WORLD_SIZE = 1;
constexpr uint32_t TIME_OUT = 1000;
constexpr uint32_t NEED_SYNC = 0;

constexpr uint32_t DEV_NUM = DIE_PER_SERVER * SERVER_NUM;

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    aclrtStream resetStream;
    aclrtContext context;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    const int64_t stride = 1; 
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, &stride, 0, 
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int ResetThreadFun(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Set current context failed, ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);

    void *resetDeviceAddr = nullptr;
    aclTensor *resetTensor = nullptr;
    std::vector<int> resetHostData(EP_WORLD_SIZE, 1);
    std::vector<int64_t> resetShape{(int64_t)EP_WORLD_SIZE};

    ret = CreateAclTensor(resetHostData, resetShape, &resetDeviceAddr, ACL_INT32, &resetTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Failed to create reset tensor.\n"); return ret);

    uint64_t resetWorkspaceSize = 0;
    aclOpExecutor *resetExecutor = nullptr;
    void *resetWorkspaceAddr = nullptr;

    ret = aclnnMoeDistributeBufferResetGetWorkspaceSize(resetTensor, hcomEpName, EP_WORLD_SIZE, NEED_SYNC, &resetWorkspaceSize, &resetExecutor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeBufferResetGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    if (resetWorkspaceSize > 0) {
        ret = aclrtMalloc(&resetWorkspaceAddr, resetWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    ret = aclnnMoeDistributeBufferReset(resetWorkspaceAddr, resetWorkspaceSize, resetExecutor, args.resetStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBufferReset failed. ret = %d \n", ret); return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.resetStream, TIME_OUT);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout reset failed. ret = %d \n", ret);
        return ret);

    if (resetWorkspaceSize > 0) {
        aclrtFree(resetWorkspaceAddr);
    }
    if (resetTensor != nullptr) {
        aclDestroyTensor(resetTensor);
    }
    if (resetDeviceAddr != nullptr) {
        aclrtFree(resetDeviceAddr);
    }
    LOG_PRINT("[INFO] Device %u finished reset\n", args.rankId);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.resetStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret: %d\n", ret); return ret);

    aclrtStream resetStream[DEV_NUM];
    aclrtContext context[DEV_NUM];

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SetDevice failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateContext(&context[rankId % DIE_PER_SERVER], rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateContext failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateStream(&resetStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create collect stream failed, ret: %d\n", ret); return ret);
    }

    int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
    }

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        uint32_t epRankId = rankId / TP_WORLD_SIZE;
        uint32_t tpRankId = rankId % TP_WORLD_SIZE;
        LOG_PRINT("[INFO] RankId %d prepare args.\n", rankId);

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].tpRankId = tpRankId;
        args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
        args[rankId].resetStream = resetStream[rankId];
        args[rankId].context = context[rankId % DIE_PER_SERVER];

        threads[rankId].reset(new(std::nothrow) std::thread(ResetThreadFun, std::ref(args[rankId])));
        CHECK_RET(threads[rankId] != nullptr, LOG_PRINT("[ERROR] Thread creation failed.\n"); return -1);
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        if (threads[rankId] && threads[rankId]->joinable()) {
            threads[rankId]->join();
        }
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        HcclCommDestroy(args[rankId].hcclEpComm);
        aclrtDestroyStream(args[rankId].resetStream);
        aclrtDestroyContext(args[rankId % DIE_PER_SERVER].context);
        aclrtResetDevice(rankId);
    }

    aclFinalize();
    LOG_PRINT("[INFO] Program finalized successfully.\n");

    return 0;
}