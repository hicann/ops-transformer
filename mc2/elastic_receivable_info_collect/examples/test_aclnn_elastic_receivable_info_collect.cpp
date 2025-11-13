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
 * \file test_aclnn_elastic_receivable_info_collect.cpp
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
#include "aclnnop/aclnn_elastic_receivable_info_collect.h"

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
constexpr uint32_t TIME_OUT = 10000;
constexpr uint32_t NEED_SYNC = 0;

constexpr uint32_t DEV_NUM = DIE_PER_SERVER * SERVER_NUM;

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclCollectComm;
    aclrtStream collectStream;
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

int DetectionThreadFun(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Set current context failed, ret: %d\n", ret); return ret);

    char hcomCollectName[128] = {0};
    ret = HcclGetCommName(args.hcclCollectComm, hcomCollectName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
    
    LOG_PRINT("[INFO] rank = %d, hcomCollectName = %s, collectStream = %p, \
                context = %p\n", args.rankId, hcomCollectName, \
                args.collectStream, args.context);

    /**************************************** 调用collect ********************************************/
    LOG_PRINT("[INFO] Device %u start collecting data\n", args.rankId);
    void *yDeviceAddr = nullptr;
    aclTensor *y = nullptr;
    std::vector<int32_t> yHostData(EP_WORLD_SIZE * EP_WORLD_SIZE, 0);
    std::vector<int64_t> yShape{(int64_t)(EP_WORLD_SIZE), (int64_t)(EP_WORLD_SIZE)};
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_INT32, &y);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Failed to create collect tensor\n"); return ret);

    uint64_t collectWorkspaceSize = 0;
    aclOpExecutor *collectExecutor = nullptr;
    void *collectWorkspaceAddr = nullptr;

    ret = aclnnElasticReceivableInfoCollectGetWorkspaceSize(hcomCollectName, EP_WORLD_SIZE, y, &collectWorkspaceSize, &collectExecutor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnCollectGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    if (collectWorkspaceSize > 0) {
        ret = aclrtMalloc(&collectWorkspaceAddr, collectWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnElasticReceivableInfoCollect(collectWorkspaceAddr, collectWorkspaceSize, collectExecutor, args.collectStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnCollect failed. ret = %d \n", ret);  \
        return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.collectStream, TIME_OUT);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout collect failed. ret = %d \n", ret);
        return ret);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    LOG_PRINT("[INFO] Device %u finished collecting data\n", args.rankId);

    // 释放device资源
    if (collectWorkspaceSize > 0) {
        aclrtFree(collectWorkspaceAddr);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }

    HcclCommDestroy(args.hcclCollectComm);
    aclrtDestroyStream(args.collectStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    LOG_PRINT("[INFO] Device %u completed all steps\n", args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret: %d\n", ret); return ret);

    aclrtStream collectStream[DEV_NUM];
    aclrtContext context[DEV_NUM];

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SetDevice failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateContext(&context[rankId % DIE_PER_SERVER], rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateContext failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateStream(&collectStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create collect stream failed, ret: %d\n", ret); return ret);
    }

    int32_t devicesCollect[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesCollect[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsCollect[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesCollect[tpId], commsCollect[tpId]);
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
        args[rankId].hcclCollectComm = commsCollect[tpRankId][epRankId];
        args[rankId].collectStream = collectStream[rankId];
        args[rankId].context = context[rankId % DIE_PER_SERVER];

        threads[rankId].reset(new(std::nothrow) std::thread(DetectionThreadFun, std::ref(args[rankId])));
        CHECK_RET(threads[rankId] != nullptr, LOG_PRINT("[ERROR] Thread creation failed.\n"); return -1);
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        if (threads[rankId] && threads[rankId]->joinable()) {
            threads[rankId]->join();
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(TIME_OUT));

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        HcclCommDestroy(args[rankId].hcclCollectComm);
        aclrtDestroyStream(args[rankId].collectStream);
        aclrtDestroyContext(args[rankId % DIE_PER_SERVER].context);
        aclrtResetDevice(rankId);
    }

    aclFinalize();
    LOG_PRINT("[INFO] Program finalized successfully.\n");

    return 0;
}