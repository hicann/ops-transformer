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
 * \file test_aclnn_kda_input_proj.cpp
 * \brief aclnnKdaInputProj 调用样例。
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_kda_input_proj.h"

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
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

// Linear.weight 存储 [N, K]，公开接口传入 matmul 右操作数 view [K, N]（stride=(1,K)），由 aclnn 推断转置。
template <typename T>
int CreateAclMatmulRhsFromNk(const std::vector<T> &hostData, int64_t nSize, int64_t kSize, void **deviceAddr,
                             aclDataType dataType, aclTensor **tensor)
{
    std::vector<int64_t> storageShape = {nSize, kSize};
    auto size = GetShapeSize(storageShape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> viewShape = {kSize, nSize};
    std::vector<int64_t> viewStrides = {1, kSize};
    *tensor =
        aclCreateTensor(viewShape.data(), static_cast<int64_t>(viewShape.size()), dataType, viewStrides.data(), 0,
                        ACL_FORMAT_ND, storageShape.data(), static_cast<int64_t>(storageShape.size()), *deviceAddr);
    return 0;
}

void DestroyTensor(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 权重按 Linear.weight [N, K] 存储；公开接口传入 [K, N] 转置 view，aclnn 按 MatMulV3 规则推断 trans=true。
    constexpr int64_t tokenNum = 1;
    constexpr int64_t hiddenSize = 256;
    constexpr int64_t qkvSize = 128;
    constexpr int64_t betaSize = 16;
    constexpr int64_t gateSize = 32;
    constexpr int64_t gSize = 32;
    constexpr int64_t mxBlock = 64;
    const int64_t mxHidden = (hiddenSize + mxBlock - 1) / mxBlock;

    std::vector<int64_t> xShape = {tokenNum, hiddenSize};
    std::vector<int64_t> weightQkvShape = {qkvSize, hiddenSize};
    std::vector<int64_t> weightBetaShape = {betaSize, hiddenSize};
    std::vector<int64_t> weightGateShape = {gateSize, hiddenSize};
    std::vector<int64_t> weightGShape = {gSize, hiddenSize};
    std::vector<int64_t> weightQkvScaleShape = {qkvSize, mxHidden, 2};
    std::vector<int64_t> qkvOutShape = {tokenNum, qkvSize};
    std::vector<int64_t> betaOutShape = {tokenNum, betaSize};
    std::vector<int64_t> gateOutShape = {tokenNum, gateSize};
    std::vector<int64_t> gOutShape = {tokenNum, gSize};

    std::vector<uint16_t> xHost(GetShapeSize(xShape), 0x3F80); // BF16 1.0
    std::vector<uint8_t> weightQkvHost(GetShapeSize(weightQkvShape), 0);
    std::vector<uint16_t> weightBetaHost(GetShapeSize(weightBetaShape), 0);
    std::vector<uint16_t> weightGateHost(GetShapeSize(weightGateShape), 0);
    std::vector<uint16_t> weightGHost(GetShapeSize(weightGShape), 0);
    std::vector<uint8_t> weightQkvScaleHost(GetShapeSize(weightQkvScaleShape), 0);
    std::vector<uint16_t> qkvOutHost(GetShapeSize(qkvOutShape), 0);
    std::vector<float> betaOutHost(GetShapeSize(betaOutShape), 0.0f);
    std::vector<uint16_t> gateOutHost(GetShapeSize(gateOutShape), 0);
    std::vector<uint16_t> gOutHost(GetShapeSize(gOutShape), 0);

    void *xDevice = nullptr;
    void *weightQkvDevice = nullptr;
    void *weightBetaDevice = nullptr;
    void *weightGateDevice = nullptr;
    void *weightGDevice = nullptr;
    void *weightQkvScaleDevice = nullptr;
    void *qkvOutDevice = nullptr;
    void *betaOutDevice = nullptr;
    void *gateOutDevice = nullptr;
    void *gOutDevice = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weightQkv = nullptr;
    aclTensor *weightBeta = nullptr;
    aclTensor *weightGate = nullptr;
    aclTensor *weightG = nullptr;
    aclTensor *weightQkvScale = nullptr;
    aclTensor *qkvOut = nullptr;
    aclTensor *betaOut = nullptr;
    aclTensor *gateOut = nullptr;
    aclTensor *gOut = nullptr;

    ret = CreateAclTensor(xHost, xShape, &xDevice, ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclMatmulRhsFromNk(weightQkvHost, qkvSize, hiddenSize, &weightQkvDevice, ACL_FLOAT8_E4M3FN, &weightQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclMatmulRhsFromNk(weightBetaHost, betaSize, hiddenSize, &weightBetaDevice, ACL_BF16, &weightBeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclMatmulRhsFromNk(weightGateHost, gateSize, hiddenSize, &weightGateDevice, ACL_BF16, &weightGate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclMatmulRhsFromNk(weightGHost, gSize, hiddenSize, &weightGDevice, ACL_BF16, &weightG);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightQkvScaleHost, weightQkvScaleShape, &weightQkvScaleDevice, ACL_FLOAT8_E8M0,
                          &weightQkvScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qkvOutHost, qkvOutShape, &qkvOutDevice, ACL_BF16, &qkvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaOutHost, betaOutShape, &betaOutDevice, ACL_FLOAT, &betaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gateOutHost, gateOutShape, &gateOutDevice, ACL_BF16, &gateOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gOutHost, gOutShape, &gOutDevice, ACL_BF16, &gOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnKdaInputProjGetWorkspaceSize(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkvOut,
                                            betaOut, gateOut, gOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKdaInputProjGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnKdaInputProj(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKdaInputProj failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(betaOutHost.data(), betaOutHost.size() * sizeof(float), betaOutDevice,
                      betaOutHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy beta from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 8 && i < static_cast<int64_t>(betaOutHost.size()); ++i) {
        LOG_PRINT("betaOut[%ld] = %f\n", i, betaOutHost[i]);
    }

    DestroyTensor(x, xDevice);
    DestroyTensor(weightQkv, weightQkvDevice);
    DestroyTensor(weightBeta, weightBetaDevice);
    DestroyTensor(weightGate, weightGateDevice);
    DestroyTensor(weightG, weightGDevice);
    DestroyTensor(weightQkvScale, weightQkvScaleDevice);
    DestroyTensor(qkvOut, qkvOutDevice);
    DestroyTensor(betaOut, betaOutDevice);
    DestroyTensor(gateOut, gateOutDevice);
    DestroyTensor(gOut, gOutDevice);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
