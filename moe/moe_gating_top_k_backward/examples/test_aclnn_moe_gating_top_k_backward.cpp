/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_aclnn_moe_gating_top_k_backward.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_backward.h"

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
    for (auto i : shape) {
        shapeSize *= i;
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
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出
    // xNorm: [M, N] = [4, 8], float32, sigmoid输出值在(0,1)之间
    std::vector<int64_t> xNormShape = {4, 8};
    std::vector<float> xNormHostData = {0.5f, 0.7f, 0.3f, 0.8f, 0.2f, 0.6f, 0.9f, 0.4f, 0.6f, 0.4f, 0.8f,
                                        0.1f, 0.7f, 0.5f, 0.3f, 0.9f, 0.2f, 0.9f, 0.5f, 0.6f, 0.8f, 0.3f,
                                        0.7f, 0.4f, 0.8f, 0.3f, 0.6f, 0.5f, 0.4f, 0.7f, 0.2f, 0.9f};
    // gradY: [M, K] = [4, 2], float32
    std::vector<int64_t> gradYShape = {4, 2};
    std::vector<float> gradYHostData = {1.0f, 0.5f, 0.8f, 0.3f, 0.6f, 0.9f, 0.4f, 0.7f};
    // expertIdx: [M, K] = [4, 2], int32
    std::vector<int64_t> expertIdxShape = {4, 2};
    std::vector<int32_t> expertIdxHostData = {3, 6, 2, 7, 1, 4, 0, 7};
    // gradX: [M, N] = [4, 8], float32
    std::vector<int64_t> gradXShape = {4, 8};
    std::vector<float> gradXHostData(32, 0);

    void *xNormDeviceAddr = nullptr;
    void *gradYDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    aclTensor *xNorm = nullptr;
    aclTensor *gradY = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *gradX = nullptr;

    ret = CreateAclTensor(xNormHostData, xNormShape, &xNormDeviceAddr, aclDataType::ACL_FLOAT, &xNorm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用aclnnMoeGatingTopKBackward
    int64_t renorm = 0;
    int64_t normType = 1; // sigmoid模式
    double routedScalingFactor = 2.5;
    double eps = 1e-20;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    ret = aclnnMoeGatingTopKBackwardGetWorkspaceSize(xNorm, gradY, expertIdx, renorm, normType, routedScalingFactor,
                                                     eps, gradX, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    ret = aclnnMoeGatingTopKBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackward failed. ERROR: %d\n", ret); return ret);

    // 4.同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值
    auto size = GetShapeSize(gradXShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradX[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor
    aclDestroyTensor(xNorm);
    aclDestroyTensor(gradY);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(gradX);

    // 7.释放device资源
    aclrtFree(xNormDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
