/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul.h"

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

// shape 基本信息
constexpr int64_t EP_WORLD_SIZE = 2;
constexpr int64_t BS = 4096;
constexpr int64_t K = 2;
constexpr int64_t H = 7168;
constexpr int64_t e = 4;
constexpr int64_t N1 = 4096;
constexpr int64_t N2 = 4096;
constexpr int64_t A = BS * K;

int LaunchOneThreadAlltoAllvQuantGroupedMatMul(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {BS * K, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmYShape = {A, N1};
    std::vector<int64_t> permuteShape = {A, H};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmYShape = {BS, N2};
    std::vector<int64_t> scaleShape = {1}; // 缩放因子形状

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmYDeviceAddr = nullptr;
    void *permuteDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmY = nullptr;
    aclTensor *permute = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;
    int64_t groupSize = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmYShapeSize = GetShapeSize(gmmYShape);
    long long permuteShapeSize = GetShapeSize(permuteShape);
    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    // HIFLOAT8数据 (使用uint8_t模拟)
    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 5);
    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 5);

    // 输出数据 (FLOAT16/BFLOAT16)
    std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
    std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);
    std::vector<uint8_t> permuteHostData(permuteShapeSize, 255);

    // 缩放因子数据 (FLOAT32)
    std::vector<float> gmmXScaleHostData(1, 1.0f);
    std::vector<float> gmmWScaleHostData(1, 1.0f);
    std::vector<float> mmXScaleHostData(1, 1.0f);
    std::vector<float> mmWScaleHostData(1, 1.0f);

    // 创建张量
    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, ACL_FLOAT16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, ACL_HIFLOAT8, &permute);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, scaleShape, &gmmXScaleDeviceAddr, ACL_FLOAT, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, scaleShape, &gmmWScaleDeviceAddr, ACL_FLOAT, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, scaleShape, &mmXScaleDeviceAddr, ACL_FLOAT, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, scaleShape, &mmWScaleDeviceAddr, ACL_FLOAT, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // 调用第一阶段接口
    ret = aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(gmmX, gmmW, gmmXScale, gmmWScale,
                                                           nullptr, // sendCountsTensorOptional
                                                           nullptr, // recvCountsTensorOptional
                                                           mmX, mmW, mmXScale, mmWScale,
                                                           1, // gmmXQuantMode
                                                           1, // gmmWeightQuantMode
                                                           1, // mmXQuantMode
                                                           1, // mmWeightQuantMode
                                                           hcomName, EP_WORLD_SIZE, sendCounts, recvCounts,
                                                           false, // transGmmWeight
                                                           false, // transMmWeight
                                                           groupSize,
                                                           true, // permuteOutFlag
                                                           gmmY, mmY, permute, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret = aclnnAlltoAllvQuantGroupedMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMul failed. ret = %d \n", ret);
              return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);

    // 释放device资源
    if (gmmX != nullptr)
        aclDestroyTensor(gmmX);
    if (gmmW != nullptr)
        aclDestroyTensor(gmmW);
    if (gmmY != nullptr)
        aclDestroyTensor(gmmY);
    if (mmX != nullptr)
        aclDestroyTensor(mmX);
    if (mmW != nullptr)
        aclDestroyTensor(mmW);
    if (mmY != nullptr)
        aclDestroyTensor(mmY);
    if (permute != nullptr)
        aclDestroyTensor(permute);
    if (gmmXScale != nullptr)
        aclDestroyTensor(gmmXScale);
    if (gmmWScale != nullptr)
        aclDestroyTensor(gmmWScale);
    if (mmXScale != nullptr)
        aclDestroyTensor(mmXScale);
    if (mmWScale != nullptr)
        aclDestroyTensor(mmWScale);

    if (gmmXDeviceAddr != nullptr)
        aclrtFree(gmmXDeviceAddr);
    if (gmmWDeviceAddr != nullptr)
        aclrtFree(gmmWDeviceAddr);
    if (gmmYDeviceAddr != nullptr)
        aclrtFree(gmmYDeviceAddr);
    if (mmXDeviceAddr != nullptr)
        aclrtFree(mmXDeviceAddr);
    if (mmWDeviceAddr != nullptr)
        aclrtFree(mmWDeviceAddr);
    if (mmYDeviceAddr != nullptr)
        aclrtFree(mmYDeviceAddr);
    if (permuteDeviceAddr != nullptr)
        aclrtFree(permuteDeviceAddr);
    if (gmmXScaleDeviceAddr != nullptr)
        aclrtFree(gmmXScaleDeviceAddr);
    if (gmmWScaleDeviceAddr != nullptr)
        aclrtFree(gmmWScaleDeviceAddr);
    if (mmXScaleDeviceAddr != nullptr)
        aclrtFree(mmXScaleDeviceAddr);
    if (mmWScaleDeviceAddr != nullptr)
        aclrtFree(mmWScaleDeviceAddr);
    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Ascend 950PR/Ascend 950DT实现
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

    constexpr uint32_t WORLD_SIZE = EP_WORLD_SIZE;
    aclrtStream stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; i++) {
        devices[i] = i;
    }

    // 初始化集合通信域
    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[WORLD_SIZE];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvQuantGroupedMatMul, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}
