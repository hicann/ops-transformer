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
 * \file test_aclnn_ffn_to_attention.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_ffn_to_attention.h"

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
    aclrtStream FFN2AttentionStream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t ATTN_NUM = 8;
constexpr uint32_t DEV_NUM = WORLD_SIZE;

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

int LaunchOneProcessFFN2Attention(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, FFN2AttentionStream = %p, \
                context = %p\n",
              args.rankId, hcomName, args.FFN2AttentionStream, args.context);

    int64_t micro_batch_num = 1;
    int64_t Y = 8;
    int64_t H = 7168;
    int64_t K = 7;
    int64_t attention_worker_num = ATTN_NUM;
    int64_t sharedExpertNum = 1;
    int64_t expert_num_per_token = K + sharedExpertNum;
    int64_t Token_info_shape[] = {micro_batch_num, Y, expert_num_per_token};
    int64_t Token_data_shape[] = {micro_batch_num, Y, expert_num_per_token, H};

    void *xDeviceAddr = nullptr;
    void *sessionIdsDeviceAddr = nullptr;
    void *microBatchIdsDeviceAddr = nullptr;
    void *tokenIdsDeviceAddr = nullptr;
    void *expertOffsetsDeviceAddr = nullptr;
    void *actualTokenNumDeviceAddr = nullptr;
    void *attnRankTableDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *sessionIds = nullptr;
    aclTensor *microBatchIds = nullptr;
    aclTensor *tokenIds = nullptr;
    aclTensor *expertOffsets = nullptr;
    aclTensor *actualTokenNum = nullptr;
    aclTensor *attnRankTable = nullptr;
    aclIntArray *tokenInfoTableShape = aclCreateIntArray(Token_info_shape, 3);
    aclIntArray *tokenDataShape = aclCreateIntArray(Token_data_shape, 4);

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{Y, H};
    std::vector<int64_t> sessionIdsShape{Y};
    std::vector<int64_t> microBatchIdsShape{Y};
    std::vector<int64_t> tokenIdsShape{Y};
    std::vector<int64_t> expertOffsetsShape{Y};
    std::vector<int64_t> actualTokenNumShape{1};
    std::vector<int64_t> attnRankTableShape{attention_worker_num};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t sessionIdsShapeSize = GetShapeSize(sessionIdsShape);
    int64_t microBatchIdsShapeSize = GetShapeSize(microBatchIdsShape);
    int64_t tokenIdsShapeSize = GetShapeSize(tokenIdsShape);
    int64_t expertOffsetsShapeSize = GetShapeSize(expertOffsetsShape);
    int64_t actualTokenNumShapeSize = GetShapeSize(actualTokenNumShape);
    int64_t attnRankTableShapeSize = GetShapeSize(attnRankTableShape);

    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> sessionIdsHostData(sessionIdsShapeSize, 0);
    std::vector<int32_t> microBatchIdsHostData(microBatchIdsShapeSize, 0);
    std::vector<int32_t> tokenIdsHostData(tokenIdsShapeSize, 0);
    std::vector<int32_t> expertOffsetsHostData(expertOffsetsShapeSize, 0);
    std::vector<int64_t> actualTokenNumHostData(actualTokenNumShapeSize, 8);
    std::vector<int32_t> attnRankTableHostData(attnRankTableShapeSize);
    for (int32_t i = 0; i < Y; i++) {
        sessionIdsHostData[i] = i % attention_worker_num;
        tokenIdsHostData[i] = i % Y;
        expertOffsetsHostData[i] = i % expert_num_per_token;
    }
    for (int32_t i = 0; i < attention_worker_num; i++) {
        attnRankTableHostData[i] = static_cast<int32_t>(i);
    }

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdsHostData, sessionIdsShape, &sessionIdsDeviceAddr, aclDataType::ACL_INT32,
                          &sessionIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdsHostData, microBatchIdsShape, &microBatchIdsDeviceAddr, aclDataType::ACL_INT32,
                          &microBatchIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tokenIdsHostData, tokenIdsShape, &tokenIdsDeviceAddr, aclDataType::ACL_INT32, &tokenIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertOffsetsHostData, expertOffsetsShape, &expertOffsetsDeviceAddr, aclDataType::ACL_INT32,
                          &expertOffsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actualTokenNumHostData, actualTokenNumShape, &actualTokenNumDeviceAddr,
                          aclDataType::ACL_INT64, &actualTokenNum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnRankTableHostData, attnRankTableShape, &attnRankTableDeviceAddr, aclDataType::ACL_INT32,
                          &attnRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t FFN2AttentionWorkspaceSize = 0;
    aclOpExecutor *FFN2AttentionExecutor = nullptr;
    void *FFN2AttentionWorkspaceAddr = nullptr;

    /**************************************** 调用FFN2Attention ********************************************/
    // 调用第一阶段接口
    ret = aclnnFFNToAttentionGetWorkspaceSize(x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
                                              attnRankTable, hcomName, WORLD_SIZE, tokenInfoTableShape, tokenDataShape,
                                              &FFN2AttentionWorkspaceSize, &FFN2AttentionExecutor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnFFNToAttentionGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (FFN2AttentionWorkspaceSize > 0) {
        ret = aclrtMalloc(&FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnFFNToAttention(FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, FFN2AttentionExecutor,
                              args.FFN2AttentionStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnFFNToAttention failed. ret = %d \n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    if (args.rankId >= ATTN_NUM) {
        ret = aclrtSynchronizeStreamWithTimeout(args.FFN2AttentionStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                  return ret);
        LOG_PRINT("[INFO] device_%d FFNToAttention execute successfully.\n", args.rankId);
    } else {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        LOG_PRINT("[INFO] device_%d is AttentionWorker, sleeping 10 seconds...\n", args.rankId);
    }

    // 释放device资源
    if (FFN2AttentionWorkspaceSize > 0) {
        aclrtFree(FFN2AttentionWorkspaceAddr);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (sessionIds != nullptr) {
        aclDestroyTensor(sessionIds);
    }
    if (microBatchIds != nullptr) {
        aclDestroyTensor(microBatchIds);
    }
    if (tokenIds != nullptr) {
        aclDestroyTensor(tokenIds);
    }

    if (expertOffsets != nullptr) {
        aclDestroyTensor(expertOffsets);
    }
    if (actualTokenNum != nullptr) {
        aclDestroyTensor(actualTokenNum);
    }
    if (attnRankTable != nullptr) {
        aclDestroyTensor(attnRankTable);
    }
    if (tokenInfoTableShape != nullptr) {
        aclDestroyIntArray(tokenInfoTableShape);
    }
    if (tokenDataShape != nullptr) {
        aclDestroyIntArray(tokenDataShape);
    }

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (sessionIdsDeviceAddr != nullptr) {
        aclrtFree(sessionIdsDeviceAddr);
    }
    if (microBatchIdsDeviceAddr != nullptr) {
        aclrtFree(microBatchIdsDeviceAddr);
    }
    if (tokenIdsDeviceAddr != nullptr) {
        aclrtFree(tokenIdsDeviceAddr);
    }
    if (expertOffsetsDeviceAddr != nullptr) {
        aclrtFree(expertOffsetsDeviceAddr);
    }
    if (actualTokenNumDeviceAddr != nullptr) {
        aclrtFree(actualTokenNumDeviceAddr);
    }
    if (attnRankTableDeviceAddr != nullptr) {
        aclrtFree(attnRankTableDeviceAddr);
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.FFN2AttentionStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

    aclrtStream FFN2AttentionStream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&FFN2AttentionStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t deviceId = 0; deviceId < WORLD_SIZE; deviceId++) {
        devices[deviceId] = deviceId;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].FFN2AttentionStream = FFN2AttentionStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcessFFN2Attention, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
