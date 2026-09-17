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
 * \file test_aclnn_mhc_pre_backward.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_backward_v2.h"

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

constexpr uint32_t NUMBER_TWO = 2U;
constexpr uint32_t BF16_TO_FP32_SHIFT = 16U;
constexpr int64_t DISPLAY_ELEMENT_COUNT = 10;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char *name, size_t elemSize = sizeof(float))
{
    auto size = GetShapeSize(shape);
    size_t copyBytes = size * elemSize;
    if (elemSize == NUMBER_TWO) {
        // BF16: read raw bytes, convert to float for display
        std::vector<uint16_t> rawData(size, 0);
        auto ret = aclrtMemcpy(rawData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, DISPLAY_ELEMENT_COUNT); i++) {
            union {
                uint32_t i;
                float f;
            } u;
            u.i = (uint32_t)rawData[i] << BF16_TO_FP32_SHIFT;
            LOG_PRINT("  [%ld] = %f\n", i, u.f);
        }
    } else {
        // float32
        std::vector<float> resultData(size, 0);
        auto ret = aclrtMemcpy(resultData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, DISPLAY_ELEMENT_COUNT); i++) {
            LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
        }
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - NUMBER_TWO; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1024, 4, 512};      // T, N, D
    std::vector<int64_t> phiShape = {24, 2048};        // 2 * N + N * N, N * D
    std::vector<int64_t> alphaShape = {3};             // 固定大小
    std::vector<int64_t> gradHInShape = {1024, 512};   // T, D
    std::vector<int64_t> gradHPostShape = {1024, 4};   // T, N
    std::vector<int64_t> gradHResShape = {1024, 4, 4}; // T, N, N
    std::vector<int64_t> invRmsShape = {1024};         // T
    std::vector<int64_t> hMixShape = {1024, 24};       // T, 2 * N + N * N
    std::vector<int64_t> hPreShape = {1024, 4};        // T, N
    std::vector<int64_t> hPostShape = {1024, 4};       // T, N
    std::vector<int64_t> gammaShape = {4, 512};        // N, D
    std::vector<int64_t> gradXShape = {1024, 4, 512};  // T, N, D
    std::vector<int64_t> gradPhiShape = {24, 2048};    // 2 * N + N * N, N * D
    std::vector<int64_t> gradAlphaShape = {3};         // 固定大小
    std::vector<int64_t> gradGammaShape = {4, 512};    // N, D
    std::vector<int64_t> gradBiasShape = {24};         // T, N, D (grad_x_post)
    std::vector<int64_t> gradXPostOptionalShape = {1024, 4, 512};

    void *xDeviceAddr = nullptr;
    void *phiDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *gradHInDeviceAddr = nullptr;
    void *gradHPostDeviceAddr = nullptr;
    void *gradHResDeviceAddr = nullptr;
    void *invRmsDeviceAddr = nullptr;
    void *hMixDeviceAddr = nullptr;
    void *hPreDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    void *gradPhiDeviceAddr = nullptr;
    void *gradAlphaDeviceAddr = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    void *gradGammaDeviceAddr = nullptr;
    void *gradXPostOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *phi = nullptr;
    aclTensor *alpha = nullptr;
    aclTensor *gradHIn = nullptr;
    aclTensor *gradHPost = nullptr;
    aclTensor *gradHRes = nullptr;
    aclTensor *invRms = nullptr;
    aclTensor *hMix = nullptr;
    aclTensor *hPre = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *gradX = nullptr;
    aclTensor *gradPhi = nullptr;
    aclTensor *gradAlpha = nullptr;
    aclTensor *gradBias = nullptr;
    aclTensor *gradGamma = nullptr;
    aclTensor *gradXPostOptional = nullptr;

    std::vector<short> xHostData(1024 * 4 * 512, 1.0);
    std::vector<float> phiHostData(24 * 2048, 1.0);
    std::vector<float> alphaHostData(3, 1.0);
    std::vector<short> gradHInHostData(1024 * 512, 1.0);
    std::vector<float> gradHPostHostData(1024 * 4, 1.0);
    std::vector<float> gradHResHostData(1024 * 4 * 4, 1.0);
    std::vector<float> invRmsHostData(1024, 1.0);
    std::vector<float> hMixHostData(1024 * 24, 1.0);
    std::vector<float> hPreHostData(1024 * 4, 1.0);
    std::vector<float> hPostHostData(1024 * 4, 1.0);
    std::vector<float> gammaHostData(4 * 512, 1.0);
    std::vector<short> gradXHostData(1024 * 4 * 512, 0);
    std::vector<float> gradPhiHostData(24 * 2048, 0);
    std::vector<float> gradAlphaHostData(3, 0);
    std::vector<float> gradBiasHostData(24, 0);
    std::vector<float> gradGammaHostData(4 * 512, 0);
    std::vector<short> gradXPostOptionalHostData(1024 * 4 * 512, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(phiHostData, phiShape, &phiDeviceAddr, aclDataType::ACL_FLOAT, &phi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHInHostData, gradHInShape, &gradHInDeviceAddr, aclDataType::ACL_BF16, &gradHIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHPostHostData, gradHPostShape, &gradHPostDeviceAddr, aclDataType::ACL_FLOAT, &gradHPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHResHostData, gradHResShape, &gradHResDeviceAddr, aclDataType::ACL_FLOAT, &gradHRes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invRmsHostData, invRmsShape, &invRmsDeviceAddr, aclDataType::ACL_FLOAT, &invRms);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hMixHostData, hMixShape, &hMixDeviceAddr, aclDataType::ACL_FLOAT, &hMix);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPreHostData, hPreShape, &hPreDeviceAddr, aclDataType::ACL_FLOAT, &hPre);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_BF16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradPhiHostData, gradPhiShape, &gradPhiDeviceAddr, aclDataType::ACL_FLOAT, &gradPhi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradAlphaHostData, gradAlphaShape, &gradAlphaDeviceAddr, aclDataType::ACL_FLOAT, &gradAlpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradGammaHostData, gradGammaShape, &gradGammaDeviceAddr, aclDataType::ACL_FLOAT, &gradGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXPostOptionalHostData, gradXPostOptionalShape, &gradXPostOptionalDeviceAddr,
                          aclDataType::ACL_BF16, &gradXPostOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float hc_eps = 1e-6;
    int64_t opImplMode = 1;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 调用aclnnMhcPreBackwardV2第一段接口
    ret = aclnnMhcPreBackwardV2GetWorkspaceSize(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost,
                                                gamma, gradXPostOptional, hc_eps, opImplMode, gradX, gradPhi, gradAlpha,
                                                gradBias, gradGamma, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackwardV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMhcPreBackwardV2第二段接口
    ret = aclnnMhcPreBackwardV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackwardV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(gradXShape, &gradXDeviceAddr, "gradX", sizeof(short));
    PrintOutResult(gradPhiShape, &gradPhiDeviceAddr, "gradPhi");
    PrintOutResult(gradAlphaShape, &gradAlphaDeviceAddr, "gradAlpha");
    PrintOutResult(gradBiasShape, &gradBiasDeviceAddr, "gradBias");
    PrintOutResult(gradGammaShape, &gradGammaDeviceAddr, "gradGamma");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(phi);
    aclDestroyTensor(alpha);
    aclDestroyTensor(gradHIn);
    aclDestroyTensor(gradHPost);
    aclDestroyTensor(gradHRes);
    aclDestroyTensor(invRms);
    aclDestroyTensor(hMix);
    aclDestroyTensor(hPre);
    aclDestroyTensor(hPost);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradX);
    aclDestroyTensor(gradPhi);
    aclDestroyTensor(gradAlpha);
    aclDestroyTensor(gradBias);
    aclDestroyTensor(gradGamma);
    aclDestroyTensor(gradXPostOptional);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(phiDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(gradHInDeviceAddr);
    aclrtFree(gradHPostDeviceAddr);
    aclrtFree(gradHResDeviceAddr);
    aclrtFree(invRmsDeviceAddr);
    aclrtFree(hMixDeviceAddr);
    aclrtFree(hPreDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradPhiDeviceAddr);
    aclrtFree(gradAlphaDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    aclrtFree(gradGammaDeviceAddr);
    aclrtFree(gradXPostOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
