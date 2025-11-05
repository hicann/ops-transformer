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
 * \file test_aclnn_elastic_receivable_test.cpp
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
#include "aclnnop/aclnn_elastic_receivable_test.h"
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
constexpr uint32_t TIME_OUT = 10000;
constexpr uint32_t NEED_SYNC = 0;

constexpr uint32_t DEV_NUM = DIE_PER_SERVER * SERVER_NUM;

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclTestComm;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream testStream;
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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("[");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%d,", resultData[i]);
    }
    LOG_PRINT("]\n");
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

static bool CheckRankValid(uint32_t rank_id, uint32_t world_size, std::unordered_set<int>&server_set, std::vector<int> rank_table_data) {
    for (int32_t i = 0; i < world_size / DIE_PER_SERVER; i++) {
        if (server_set.count(i) > 0) {
            continue;
        }
        for (int32_t j = 0; j < DIE_PER_SERVER; j++) {
            if (rank_table_data[rank_id * world_size + i * DIE_PER_SERVER + j] == 0) {
                return false;
            }
        }
    }
    for (int32_t i = 0; i < world_size / DIE_PER_SERVER; i++) {
        if (server_set.count(i) > 0) {
            continue;
        }
        for (int32_t j = 0; j < DIE_PER_SERVER; j++) {
            int32_t offset = 0;
            offset = ((i * DIE_PER_SERVER + j)) * world_size;
            if (rank_table_data[offset + rank_id] == 0) {
                return false;
            }
        }
    }
    return true;
}

static bool CheckRowIsZeros(uint32_t rank_id, uint32_t world_size, std::unordered_set<int>&server_set, std::vector<int> rank_table_data) {
    for (int32_t i = 0; i < world_size / DIE_PER_SERVER; i++) {
        if (server_set.count(i) > 0) {
            continue;
        }
        for (int32_t j = 0; j < DIE_PER_SERVER; j++) {
            if (rank_table_data[rank_id * world_size + i * DIE_PER_SERVER + j] > 0) {
                return false;
            }
        }
    }
    return true;
}

static bool CalValidRankListInfo(int32_t rankId, std::vector<int> rank_table_data, int64_t world_size, int *output_elastic_rank) { 
    std::unordered_set<int>server_set;
    for (int rank_id = 0; rank_id < world_size; rank_id++) {
        if (CheckRowIsZeros(rank_id, world_size, server_set, rank_table_data)) {
            server_set.insert(rank_id / DIE_PER_SERVER);
        }
    }
    if (server_set.size() == (world_size / DIE_PER_SERVER)) {
        for (int i = 0; i < world_size; i++) {
            output_elastic_rank[i] = 0;
        }
        return false;
    }
    for (int rank_id = 0; rank_id < world_size; rank_id++) {
        bool is_valid = CheckRankValid(rank_id, world_size, server_set, rank_table_data);
        if (is_valid) {
            output_elastic_rank[rank_id] = 1;
        } else {
            // roll back
            server_set.insert(rank_id / DIE_PER_SERVER);
            uint32_t begin_rank = rank_id / DIE_PER_SERVER * DIE_PER_SERVER;
            for (int i = begin_rank; i < rank_id; i++) {
                output_elastic_rank[i] = 0;
            }
            rank_id = begin_rank + DIE_PER_SERVER - 1;
        }
    }
    std::cout << "rank_table_data vector print: ";
    for (size_t i = 0; i < rank_table_data.size(); ++i) {
        std::cout << rank_table_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "output_elastic_rank: ";
    for (int i = 0; i < EP_WORLD_SIZE; ++i) {
        std::cout << *(output_elastic_rank + i) << " ";
    }
    std::cout << std::endl;
    return output_elastic_rank[rankId] == 1;
}

int DetectionThreadFun(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Set current context failed, ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
    char hcomTestName[128] = {0};
    ret = HcclGetCommName(args.hcclTestComm, hcomTestName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
    
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, hcomTestName = %s, testStream = %p, collectStream = %p, \
                context = %p\n", args.rankId, hcomEpName, hcomTpName, hcomTestName, args.testStream, \
                args.collectStream, args.context);

    /**************************************** 每个server调用test ********************************************/
    for (int32_t server = 0; server < EP_WORLD_SIZE / DIE_PER_SERVER; ++server) {
        void *destRankDeviceAddr = nullptr;
        aclTensor *destRank = nullptr;
        std::vector<int32_t> destRankHostData(DIE_PER_SERVER);
        std::iota(destRankHostData.begin(), destRankHostData.end(), server * DIE_PER_SERVER);
        
        std::vector<int64_t> destRankShape{(int64_t)DIE_PER_SERVER};
        ret = CreateAclTensor(destRankHostData, destRankShape, &destRankDeviceAddr, ACL_INT32, &destRank);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Failed to create tensor\n"); return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        ret = aclnnElasticReceivableTestGetWorkspaceSize(destRank, hcomTestName, EP_WORLD_SIZE, DIE_PER_SERVER, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnTestGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnElasticReceivableTest(workspaceAddr, workspaceSize, executor, args.testStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnTest failed. ret = %d \n", ret);  \
            return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.testStream, TIME_OUT);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        PrintOutResult(destRankShape, &destRankDeviceAddr);
        // 释放device资源
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        if (destRank != nullptr) {
            aclDestroyTensor(destRank);
        }
        if (destRankDeviceAddr != nullptr) {
            aclrtFree(destRankDeviceAddr);
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(TIME_OUT));

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

    ret = aclnnElasticReceivableInfoCollectGetWorkspaceSize(hcomTestName, EP_WORLD_SIZE, y, &collectWorkspaceSize, &collectExecutor);
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
    std::vector<int> rank_table_data(EP_WORLD_SIZE * EP_WORLD_SIZE, 0);
    ret = aclrtMemcpy(rank_table_data.data(), rank_table_data.size() * sizeof(rank_table_data[0]), yDeviceAddr,
                        EP_WORLD_SIZE * EP_WORLD_SIZE * sizeof(rank_table_data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::vector<int> output_elastic_rank(EP_WORLD_SIZE, 0);
    int* output_elastic_rank_ptr = output_elastic_rank.data();
    
    bool needClean = CalValidRankListInfo(1, rank_table_data, EP_WORLD_SIZE, std::ref(output_elastic_rank_ptr));
    if (needClean) {
        void *resetDeviceAddr = nullptr;
        aclTensor *resetTensor = nullptr;
        std::vector<int> resetHostData(EP_WORLD_SIZE, 0);
        std::copy(output_elastic_rank_ptr, output_elastic_rank_ptr + EP_WORLD_SIZE, resetHostData.begin());
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
        ret = aclnnMoeDistributeBufferReset(resetWorkspaceAddr, resetWorkspaceSize, resetExecutor, args.collectStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBufferReset failed. ret = %d \n", ret); return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.collectStream, TIME_OUT);
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
    }

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
    
    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    HcclCommDestroy(args.hcclTestComm);
    aclrtDestroyStream(args.testStream);
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

    aclrtStream testStream[DEV_NUM];
    aclrtStream collectStream[DEV_NUM];
    aclrtContext context[DEV_NUM];

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SetDevice failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateContext(&context[rankId % DIE_PER_SERVER], rankId % DIE_PER_SERVER);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateContext failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateStream(&testStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create test stream failed, ret: %d\n", ret); return ret);

        ret = aclrtCreateStream(&collectStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create collect stream failed, ret: %d\n", ret); return ret);
    }

    int32_t devicesTest[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesTest[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsTest[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesTest[tpId], commsTest[tpId]);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
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

    int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
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
        args[rankId].hcclTestComm = commsTest[tpRankId][epRankId];
        args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
        args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
        args[rankId].testStream = testStream[rankId];
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
        HcclCommDestroy(args[rankId].hcclEpComm);
        HcclCommDestroy(args[rankId].hcclTpComm);
        HcclCommDestroy(args[rankId].hcclTestComm);
        aclrtDestroyStream(args[rankId].testStream);
        aclrtDestroyStream(args[rankId].collectStream);
        aclrtDestroyContext(args[rankId % DIE_PER_SERVER].context);
        aclrtResetDevice(rankId);
    }

    aclFinalize();
    LOG_PRINT("[INFO] Program finalized successfully.\n");

    return 0;
}