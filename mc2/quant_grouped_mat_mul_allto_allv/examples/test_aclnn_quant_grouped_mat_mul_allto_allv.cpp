/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_grouped_mat_mul_allto_allv.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_quant_grouped_mat_mul_allto_allv.h"

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

std::vector<int16_t> pGmmYData(BS *K *N1, 0);
std::vector<int16_t> pmmYData(BS *N2, 0);

int LaunchOneThreadAlltoAllvGmm(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {A, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmXScaleShape = {1};
    std::vector<int64_t> gmmWScaleShape = {1};
    std::vector<int64_t> yShape = {BS * K, N1};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmXScaleShape = {1};
    std::vector<int64_t> mmWScaleShape = {1};
    std::vector<int64_t> mmYShape = {BS, N2};

    std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
    std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *y = nullptr;

    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;
    aclTensor *mmY = nullptr;

    aclTensor *sendCountsTensor = nullptr;
    aclTensor *recvCountsTensor = nullptr;
    aclTensor *commQuantScaleOptional = nullptr;

    int64_t gmmXQuantMode = 1;
    int64_t gmmWQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWQuantMode = 1;
    int64_t commQuantMode = 0;
    int64_t commQuantDtypeOptional = -1;
    int64_t groupSize = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmXScaleShapeSize = GetShapeSize(gmmXScaleShape);
    long long gmmWScaleShapeSize = GetShapeSize(gmmWScaleShape);
    long long yShapeSize = GetShapeSize(yShape);

    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmXScaleShapeSize = GetShapeSize(mmXScaleShape);
    long long mmWScaleShapeSize = GetShapeSize(mmWScaleShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024); // HIFLOAT8
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
    std::vector<float> gmmXScaleHostData(gmmXScaleShapeSize, 1);
    std::vector<float> gmmWScaleHostData(gmmWScaleShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 65535);

    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024); // HIFLOAT8
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
    std::vector<float> mmXScaleHostData(mmXScaleShapeSize, 1);
    std::vector<float> mmWScaleHostData(mmWScaleShapeSize, 1);
    std::vector<int16_t> mmYHostData(mmYShapeSize, 0);

    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, gmmXScaleShape, &gmmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, gmmWScaleShape, &gmmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, mmXScaleShape, &mmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, mmWScaleShape, &mmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // 调用第一阶段接口
    ret = aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmW, gmmXScale, gmmWScale, sendCountsTensor, recvCountsTensor, mmX, mmW, mmXScale, mmWScale,
        commQuantScaleOptional, gmmXQuantMode, gmmWQuantMode, mmXQuantMode, mmWQuantMode, commQuantMode,
        commQuantDtypeOptional, groupSize, hcomName, EP_WORLD_SIZE, sendCounts, recvCounts, false, false, y, mmY,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnQuantGroupedMatMulAlltoAllv(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllv failed. ret = %d \n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantGroupedMatMulAlltoAllv execute successfully.\n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (args.rankId == 0) {
        size_t size = A * N1 * sizeof(int16_t);
        aclrtMemcpy(pGmmYData.data(), size, yDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (gmmX != nullptr) {
        aclDestroyTensor(gmmX);
    }
    if (gmmW != nullptr) {
        aclDestroyTensor(gmmW);
    }
    if (gmmXScale != nullptr) {
        aclDestroyTensor(gmmXScale);
    }
    if (gmmWScale != nullptr) {
        aclDestroyTensor(gmmWScale);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (mmX != nullptr) {
        aclDestroyTensor(mmX);
    }
    if (mmW != nullptr) {
        aclDestroyTensor(mmW);
    }
    if (mmXScale != nullptr) {
        aclDestroyTensor(mmXScale);
    }
    if (mmWScale != nullptr) {
        aclDestroyTensor(mmWScale);
    }
    if (mmY != nullptr) {
        aclDestroyTensor(mmY);
    }
    if (gmmXDeviceAddr != nullptr) {
        aclrtFree(gmmXDeviceAddr);
    }
    if (gmmWDeviceAddr != nullptr) {
        aclrtFree(gmmWDeviceAddr);
    }
    if (gmmXScaleDeviceAddr != nullptr) {
        aclrtFree(gmmXScaleDeviceAddr);
    }
    if (gmmWScaleDeviceAddr != nullptr) {
        aclrtFree(gmmWScaleDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (mmXDeviceAddr != nullptr) {
        aclrtFree(mmXDeviceAddr);
    }
    if (mmWDeviceAddr != nullptr) {
        aclrtFree(mmWDeviceAddr);
    }
    if (mmXScaleDeviceAddr != nullptr) {
        aclrtFree(mmXScaleDeviceAddr);
    }
    if (mmWScaleDeviceAddr != nullptr) {
        aclrtFree(mmWScaleDeviceAddr);
    }
    if (mmYDeviceAddr != nullptr) {
        aclrtFree(mmYDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[EP_WORLD_SIZE];
    aclrtContext context[EP_WORLD_SIZE];
    for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[EP_WORLD_SIZE];
    for (int i = 0; i < EP_WORLD_SIZE; i++) {
        devices[i] = i;
    }
    // 初始化集合通信域
    HcclComm comms[EP_WORLD_SIZE];
    ret = HcclCommInitAll(EP_WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[EP_WORLD_SIZE];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(EP_WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvGmm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
