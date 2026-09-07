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
 * \file test_aclnn_attention_to_ffn.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_attention_to_ffn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream attentionToFFNStream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t FFN_WORKER_NUM = 5;
constexpr uint32_t ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int LaunchOneProcessAttentionToFFN(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, attentionToFFNStream = %p, context = %p\n", args.rankId, hcomName,
              args.attentionToFFNStream, args.context);

    int64_t X = 1;
    int64_t L = 1;
    int64_t Bs = 8;
    int64_t H = 7168;
    int64_t K = 4;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 8;
    int64_t expertNumPerToken = K + sharedExpertNum;
    int64_t M = 2 * (FFN_WORKER_NUM - sharedExpertNum * sharedExpertRankNum) + 1;
    int64_t quantMode = 0;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t ffnTokenInfoTableShapeData[] = {ATTENTION_WORKER_NUM, X, 2 + Bs * expertNumPerToken};
    int64_t ffnTokenDataShapeData[] = {ATTENTION_WORKER_NUM, X, Bs, expertNumPerToken, H};
    int64_t attnTokenInfoTableShapeData[] = {X, Bs, expertNumPerToken};

    /* 根据当前场景，构造device侧输入输出变量 */
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *sessionIdDeviceAddr = nullptr;
    void *microBatchIdDeviceAddr = nullptr;
    void *layerIdDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertRankTableDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *sessionId = nullptr;
    aclTensor *microBatchId = nullptr;
    aclTensor *layerId = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertRankTable = nullptr;
    aclTensor *scales = nullptr;
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTableShapeData, 3);
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenDataShapeData, 5);
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTableShapeData, 3);

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{X, Bs, H};
    std::vector<int64_t> sessionIdShape{X};
    std::vector<int64_t> microBatchIdShape{X};
    std::vector<int64_t> layerIdShape{X};
    std::vector<int64_t> expertIdsShape{X, Bs, K};
    std::vector<int64_t> expertRankTableShape{L, moeExpertNum + sharedExpertNum, M};
    std::vector<int64_t> scalesShape{L, moeExpertNum + sharedExpertNum, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t sessionIdShapeSize = GetShapeSize(sessionIdShape);
    int64_t microBatchIdShapeSize = GetShapeSize(microBatchIdShape);
    int64_t layerIdShapeSize = GetShapeSize(layerIdShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t expertRankTableShapeSize = GetShapeSize(expertRankTableShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);

    // 构造host侧变量
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> sessionIdHostData(sessionIdShapeSize, args.rankId - FFN_WORKER_NUM);
    std::vector<int16_t> microBatchIdHostData(microBatchIdShapeSize, 0);
    std::vector<int16_t> layerIdHostData(layerIdShapeSize, 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t micro_batch_id = 0; micro_batch_id < expertIdsShape[0]; micro_batch_id++) {
        for (int32_t token_id = 0; token_id < expertIdsShape[1]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[2]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
    }

    std::vector<int32_t> expertRankTableHostData = {4, 2, 4, 3, 7, 1, 3, 2, 5, 2, 2, 5, 1, 2, 0, 0, 0, 0, 3, 2, 5,
                                                    0, 0, 3, 7, 0, 0, 4, 1, 3, 0, 1, 2, 4, 3, 7, 4, 0, 0, 3, 6, 1,
                                                    3, 2, 5, 3, 3, 7, 2, 4, 1, 2, 0, 0, 2, 2, 5, 0, 0, 0, 0, 0, 0,
                                                    3, 3, 6, 2, 5, 3, 7, 0, 0, 1, 4, 8, 0, 0, 0, 0, 0, 0};

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdHostData, sessionIdShape, &sessionIdDeviceAddr, aclDataType::ACL_INT32, &sessionId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdHostData, microBatchIdShape, &microBatchIdDeviceAddr, aclDataType::ACL_INT32,
                          &microBatchId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(layerIdHostData, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertRankTableHostData, expertRankTableShape, &expertRankTableDeviceAddr,
                          aclDataType::ACL_INT32, &expertRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t attentionToFFNWorkspaceSize = 0;
    aclOpExecutor *attentionToFFNExecutor = nullptr;
    void *attentionToFFNWorkspaceAddr = nullptr;

    /**************************************** 调用AttentionToFFN ********************************************/
    // 调用第一阶段接口
    ret = aclnnAttentionToFFNGetWorkspaceSize(
        x, sessionId, microBatchId, layerId, expertIds, expertRankTable, (quantMode > 0 ? scales : nullptr), nullptr,
        hcomName, WORLD_SIZE, ffnTokenInfoTableShape, ffnTokenDataShape, attnTokenInfoTableShape, moeExpertNum,
        quantMode, syncFlag, ffnStartRankId, &attentionToFFNWorkspaceSize, &attentionToFFNExecutor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAttentionToFFNGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (attentionToFFNWorkspaceSize > 0) {
        ret = aclrtMalloc(&attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    if (args.rankId < FFN_WORKER_NUM) { // FFN Worker
        // 等待 Attention Worker 任务执行结束
        LOG_PRINT("[INFO] device_%d is FFN worker, skipping aclnnAttentionToFFN execute.\n", args.rankId);
        std::this_thread::sleep_for(std::chrono::seconds(30));
    } else { // Attention Worker
        // 调用第二阶段接口
        ret = aclnnAttentionToFFN(attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize, attentionToFFNExecutor,
                                  args.attentionToFFNStream);

        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.attentionToFFNStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAttentionToFFN failed. ret = %d \n", ret); return ret);

        LOG_PRINT("[INFO] device_%d aclnnAttentionToFFN execute successfully.\n", args.rankId);
    }

    // 释放device资源
    if (attentionToFFNWorkspaceSize > 0) {
        aclrtFree(attentionToFFNWorkspaceAddr);
    }

    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (sessionId != nullptr) {
        aclDestroyTensor(sessionId);
    }
    if (microBatchId != nullptr) {
        aclDestroyTensor(microBatchId);
    }
    if (layerId != nullptr) {
        aclDestroyTensor(layerId);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expertRankTable != nullptr) {
        aclDestroyTensor(expertRankTable);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (ffnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(ffnTokenInfoTableShape);
    }
    if (ffnTokenDataShape != nullptr) {
        aclDestroyIntArray(ffnTokenDataShape);
    }
    if (attnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(attnTokenInfoTableShape);
    }

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (sessionIdDeviceAddr != nullptr) {
        aclrtFree(sessionIdDeviceAddr);
    }
    if (microBatchIdDeviceAddr != nullptr) {
        aclrtFree(microBatchIdDeviceAddr);
    }
    if (layerIdDeviceAddr != nullptr) {
        aclrtFree(layerIdDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expertRankTableDeviceAddr != nullptr) {
        aclrtFree(expertRankTableDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.attentionToFFNStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Atlas A3实现，必须在Atlas A3上运行
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

    aclrtStream attentionToFFNStream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&attentionToFFNStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t id = 0; id < WORLD_SIZE; id++) {
        devices[id] = id;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);

    Args args[WORLD_SIZE];
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].attentionToFFNStream = attentionToFFNStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcessAttentionToFFN, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
