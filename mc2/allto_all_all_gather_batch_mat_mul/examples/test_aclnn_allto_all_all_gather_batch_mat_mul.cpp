/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file test_aclnn_allto_all_all_gather_batch_mat_mul.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "../op_host/op_api/aclnn_all_to_all_all_gather_batch_matmul.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

constexpr int EP_WORLD_SIZE = 4;
constexpr int TP_WORLD_SIZE = 2;
constexpr int DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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

struct Args {
    int rankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadAlltoAllAllGatherBmm(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
              hcomEpName, hcomTpName, args.stream, args.context);

    int64_t E = 2 * EP_WORLD_SIZE;
    int64_t C = 2 * TP_WORLD_SIZE;
    int64_t H = 6 * TP_WORLD_SIZE;
    int64_t M = 6 * TP_WORLD_SIZE;
    int64_t xShardType = 1; // 可切换为0，使能gather H 轴场景
    int64_t actType = 1;

    std::vector<int64_t> xShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> y1OutShape;
    std::vector<int64_t> y2OutShape;
    std::vector<int64_t> y3OutShape;

    if (xShardType == 1) {
        xShape = {E, C / TP_WORLD_SIZE, H};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
    } else if (xShardType == 0) {
        xShape = {E, C, H / TP_WORLD_SIZE};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
    } else {
        LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
        return -1;
    }

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *y1OutDeviceAddr = nullptr;
    void *y2OutDeviceAddr = nullptr;
    void *y3OutDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *y1Out = nullptr;
    aclTensor *y2Out = nullptr;
    aclTensor *y3Out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    long long weightShapeSize = GetShapeSize(weightShape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long y1OutShapeSize = GetShapeSize(y1OutShape);
    long long y2OutShapeSize = GetShapeSize(y2OutShape);
    long long y3OutShapeSize = GetShapeSize(y3OutShape);

    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> weightHostData(weightShapeSize, 2);
    std::vector<int16_t> biasHostData(biasShapeSize, 3);
    std::vector<int16_t> y1OutHostData(y1OutShapeSize, 0);
    std::vector<int16_t> y2OutHostData(y2OutShapeSize, 0);
    std::vector<int16_t> y3OutHostData(y3OutShapeSize, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT16, &y1Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT16, &y2Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y3OutHostData, y3OutShape, &y3OutDeviceAddr, aclDataType::ACL_FLOAT16, &y3Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
                                                            TP_WORLD_SIZE, xShardType, actType, y1Out, y2Out, y3Out,
                                                            &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnAlltoAllAllGatherBatchMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMul failed. ret = %d \n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnAlltoAllAllGatherBatchMatMul execute successfully.\n", args.rankId);
    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (y1Out != nullptr) {
        aclDestroyTensor(y1Out);
    }
    if (y2Out != nullptr) {
        aclDestroyTensor(y2Out);
    }
    if (y3Out != nullptr) {
        aclDestroyTensor(y3Out);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (weightDeviceAddr != nullptr) {
        aclrtFree(weightDeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (y1OutDeviceAddr != nullptr) {
        aclrtFree(y1OutDeviceAddr);
    }
    if (y2OutDeviceAddr != nullptr) {
        aclrtFree(y2OutDeviceAddr);
    }
    if (y3OutDeviceAddr != nullptr) {
        aclrtFree(y3OutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devicesEp[DEV_NUM];
    int32_t devicesTp[DEV_NUM];

    // 初始化ep域  ep = 4  {0,2,4,6} {1,3,5,7}
    HcclComm commsEp[DEV_NUM];
    for (int i = 0; i < TP_WORLD_SIZE; i++) {
        for (int j = 0; j < EP_WORLD_SIZE; j++) {
            devicesEp[j + i * EP_WORLD_SIZE] = i + j * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(EP_WORLD_SIZE, &devicesEp[i * EP_WORLD_SIZE], &commsEp[i * EP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
                  return ret);
    }

    // 初始化tp域  tp = 4  {0,1},{2,3},{4,5},{6,7}
    HcclComm commsTp[DEV_NUM];
    for (int i = 0; i < EP_WORLD_SIZE; i++) {
        for (int j = 0; j < TP_WORLD_SIZE; j++) {
            devicesTp[j + i * TP_WORLD_SIZE] = j + i * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(TP_WORLD_SIZE, &devicesTp[i * TP_WORLD_SIZE], &commsTp[i * TP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
                  return ret);
    }

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = commsEp[rankId % TP_WORLD_SIZE * EP_WORLD_SIZE + rankId / TP_WORLD_SIZE];
        // Tp init按照顺序init，但同域是device id配对为[0,4],[1,5],...,[3,7].
        args[rankId].hcclTpComm = commsTp[rankId];
        std::cout << "test devices id " << rankId << " = " << devicesTp[rankId] << std::endl;
        if (rankId == (DEV_NUM - 1)) {
            args[rankId].hcclTpComm = commsTp[rankId];
        }
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllAllGatherBmm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}