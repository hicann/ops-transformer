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
 * \file test_aclnn_allto_allv_grouped_mat_mul.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "../op_host/op_api/aclnn_allto_allv_grouped_mat_mul.h"

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
constexpr int64_t EP_WORLD_SIZE = 8;
constexpr int64_t BS = 4096;
constexpr int64_t K = 2;
constexpr int64_t H = 7168;
constexpr int64_t e = 4;
constexpr int64_t N1 = 4096;
constexpr int64_t N2 = 4096;
constexpr int64_t A = BS * K;

std::vector<int16_t> pPermuteData(A *H, 0);
std::vector<int16_t> pGmmyData(A *N1, 0);
std::vector<int16_t> pmmXData(BS *H, 0);
std::vector<int16_t> pmmWData(H *N2, 0);
std::vector<int16_t> pmmYData(BS *N2, 0);

int LaunchOneThreadAlltoAllvGmm(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {BS * K, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmYShape = {A, N1};
    std::vector<int64_t> permuteShape = {A, H};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmYShape = {BS, N2};
    std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
    std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmYDeviceAddr = nullptr;
    void *permuteDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmY = nullptr;
    aclTensor *permute = nullptr;
    aclTensor *sendCountsTensor = nullptr;
    aclTensor *recvCountsTensor = nullptr;

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

    std::vector<uint16_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024); // BF16, FP16
    std::vector<uint16_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
    std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
    std::vector<uint16_t> permuteHostData(permuteShapeSize, 65535);
    std::vector<uint16_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024); // BF16, FP16
    std::vector<uint16_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
    std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);

    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_FLOAT16, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_FLOAT16, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, aclDataType::ACL_FLOAT16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_FLOAT16, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_FLOAT16, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, aclDataType::ACL_FLOAT16, &permute);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // 调用第一阶段接口
    ret = aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(gmmX, gmmW, sendCountsTensor, recvCountsTensor, mmX, mmW,
                                                      hcomName, EP_WORLD_SIZE, sendCounts, recvCounts, false, false,
                                                      true, gmmY, mmY, permute, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnAlltoAllvGroupedMatMulGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret = aclnnAlltoAllvGroupedMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvGroupedMatMul failed. ret = %d \n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnAlltoAllvGroupedMatMul execute successfully.\n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (args.rankId == 0) {
        size_t size = A * H * sizeof(int16_t);
        aclrtMemcpy(pPermuteData.data(), size, permuteDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (args.rankId == 0) {
        size_t size = A * N1 * sizeof(int16_t);
        aclrtMemcpy(pGmmyData.data(), size, gmmYDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (gmmX != nullptr) {
        aclDestroyTensor(gmmX);
    }
    if (gmmW != nullptr) {
        aclDestroyTensor(gmmW);
    }
    if (gmmY != nullptr) {
        aclDestroyTensor(gmmY);
    }
    if (mmX != nullptr) {
        aclDestroyTensor(mmX);
    }
    if (mmW != nullptr) {
        aclDestroyTensor(mmW);
    }
    if (mmY != nullptr) {
        aclDestroyTensor(mmY);
    }
    if (permute != nullptr) {
        aclDestroyTensor(permute);
    }
    if (gmmXDeviceAddr != nullptr) {
        aclrtFree(gmmXDeviceAddr);
    }
    if (gmmWDeviceAddr != nullptr) {
        aclrtFree(gmmWDeviceAddr);
    }
    if (gmmYDeviceAddr != nullptr) {
        aclrtFree(gmmYDeviceAddr);
    }
    if (mmXDeviceAddr != nullptr) {
        aclrtFree(mmXDeviceAddr);
    }
    if (mmWDeviceAddr != nullptr) {
        aclrtFree(mmWDeviceAddr);
    }
    if (mmYDeviceAddr != nullptr) {
        aclrtFree(mmYDeviceAddr);
    }
    if (permuteDeviceAddr != nullptr) {
        aclrtFree(permuteDeviceAddr);
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