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
 * \file test_aclnn_matmul_reduce_scatter_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <cstring>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_matmul_reduce_scatter_v2.h"

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

constexpr int DEV_NUM = 2;
constexpr const char *COMM_MODE = "ccu";

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
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadMmReduceScatterV2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    std::vector<int64_t> x1Shape = {1024, 256};
    std::vector<int64_t> x2Shape = {256, 512};
    std::vector<int64_t> biasShape = {512};
    std::vector<int64_t> x1ScaleShape = {1};
    std::vector<int64_t> x2ScaleShape = {1};
    std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x1ScaleDeviceAddr = nullptr;
    void *x2ScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *x1Scale = nullptr;
    aclTensor *x2Scale = nullptr;
    aclTensor *quantScale = nullptr;
    aclTensor *out = nullptr;
    aclTensor *amaxOut = nullptr;

    int32_t commTurn = 0;
    int32_t streamMode = 1;
    int32_t blockSize = 0;
    int32_t groupSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
    long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
    long long outShapeSize = GetShapeSize(outShape);

    std::vector<int8_t> x1HostData(x1ShapeSize, 0);
    std::vector<int8_t> x2HostData(x2ShapeSize, 0);
    std::vector<int32_t> biasHostData(biasShapeSize, 0);
    std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 0);
    std::vector<int32_t> x2ScaleHostData(x2ScaleShapeSize, 0);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    // 创建tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName,
                                                     "sum", commTurn, streamMode, groupSize, COMM_MODE, out, amaxOut,
                                                     &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2GetWorkspaceSize failed. ret = %d \n", ret);
              return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (x1Scale != nullptr) {
        aclDestroyTensor(x1Scale);
    }
    if (x2Scale != nullptr) {
        aclDestroyTensor(x2Scale);
    }
    if (quantScale != nullptr) {
        aclDestroyTensor(quantScale);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (amaxOut != nullptr) {
        aclDestroyTensor(amaxOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (x1ScaleDeviceAddr != nullptr) {
        aclrtFree(x1ScaleDeviceAddr);
    }
    if (x2ScaleDeviceAddr != nullptr) {
        aclrtFree(x2ScaleDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    class EnvGuard {
    public:
        EnvGuard(const char *key, const char *val, bool enable = true)
            : key_(enable ? key : nullptr)
        {
            if (!enable)
                return;
            const char *old = getenv(key);
            if (old != nullptr) {
                strncpy(saved_, old, sizeof(saved_) - 1);
            }
            setenv(key, val, 1);
        }
        ~EnvGuard()
        {
            if (key_ == nullptr)
                return;
            if (saved_[0] != '\0') {
                setenv(key_, saved_, 1);
            } else {
                unsetenv(key_);
            }
        }

    private:
        const char *key_;
        char saved_[256] = {0};
    };

    EnvGuard envGuard("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", strcmp(COMM_MODE, "ccu") == 0);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    // 初始化集合通信域
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&LaunchOneThreadMmReduceScatterV2, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
