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
 * \file test_aclnn_fused_floyd_attention_grad.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_floyd_attention_grad.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t B = 1;
    int64_t N = 1;
    int64_t S1 = 256;
    int64_t S2 = 256;
    int64_t S3 = 256;
    int64_t D = 128;

    int64_t q_size = B * N * S1 * S2 * D;
    int64_t k1_v1_size = B * N * S1 * S3 * D;
    int64_t k2_v2_size = B * N * S3 * S2 * D;
    int64_t atten_mask_size = B * S1 * S3;
    int64_t softmax_size = B * N * S1 * S2 * 8;

    std::vector<int64_t> qShape = {B, N, S1, S2, D};
    std::vector<int64_t> k1v1Shape = {B, N, S1, S3, D};
    std::vector<int64_t> k2v2Shape = {B, N, S3, S2, D};
    std::vector<int64_t> attenmaskShape = {B, 1, S1, 1, S3};
    std::vector<int64_t> softmaxMaxShape = {B, N, S1, S2, 8};
    std::vector<int64_t> softmaxSumShape = {B, N, S1, S2, 8};
    std::vector<int64_t> attentionInShape = {B, N, S1, S2, D};

    std::vector<int64_t> dqShape = {B, N, S1, S2, D};
    std::vector<int64_t> dk1dv1Shape = {B, N, S1, S3, D};
    std::vector<int64_t> dk2dv2Shape = {B, N, S3, S2, D};

    void* qDeviceAddr = nullptr;
    void* k1DeviceAddr = nullptr;
    void* v1DeviceAddr = nullptr;
    void* k2DeviceAddr = nullptr;
    void* v2DeviceAddr = nullptr;
    void* dxDeviceAddr = nullptr;
    void* attenmaskDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* attentionInDeviceAddr = nullptr;
    void* dqDeviceAddr = nullptr;
    void* dk1DeviceAddr = nullptr;
    void* dv1DeviceAddr = nullptr;
    void* dk2DeviceAddr = nullptr;
    void* dv2DeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k1 = nullptr;
    aclTensor* v1 = nullptr;
    aclTensor* k2 = nullptr;
    aclTensor* v2 = nullptr;
    aclTensor* dx = nullptr;
    aclTensor* attenmask = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* attentionIn = nullptr;
    aclTensor* dq = nullptr;
    aclTensor* dk1 = nullptr;
    aclTensor* dv1 = nullptr;
    aclTensor* dk2 = nullptr;
    aclTensor* dv2 = nullptr;

    std::vector<float> qHostData(q_size, 1.0);
    std::vector<float> k1HostData(k1_v1_size, 1.0);
    std::vector<float> v1HostData(k1_v1_size, 1.0);
    std::vector<float> k2HostData(k2_v2_size, 1.0);
    std::vector<float> v2HostData(k2_v2_size, 1.0);
    std::vector<float> dxHostData(q_size, 1.0);
    std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
    std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
    std::vector<float> softmaxSumHostData(softmax_size, 3.0);
    std::vector<float> attentionInHostData(q_size, 1.0);
    std::vector<float> dqHostData(q_size, 0);
    std::vector<float> dk1HostData(k1_v1_size, 0);
    std::vector<float> dv1HostData(k1_v1_size, 0);
    std::vector<float> dk2HostData(k2_v2_size, 0);
    std::vector<float> dv2HostData(k2_v2_size, 0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(k1HostData, k1v1Shape, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(v1HostData, k1v1Shape, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(k2HostData, k2v2Shape, &k2DeviceAddr, aclDataType::ACL_FLOAT16, &k2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(v2HostData, k2v2Shape, &v2DeviceAddr, aclDataType::ACL_FLOAT16, &v2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dxHostData, qShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dk1HostData, dk1dv1Shape, &dk1DeviceAddr, aclDataType::ACL_FLOAT16, &dk1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dv1HostData, dk1dv1Shape, &dv1DeviceAddr, aclDataType::ACL_FLOAT16, &dv1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dk2HostData, dk2dv2Shape, &dk2DeviceAddr, aclDataType::ACL_FLOAT16, &dk2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dv2HostData, dk2dv2Shape, &dv2DeviceAddr, aclDataType::ACL_FLOAT16, &dv2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double scaleValue = 1.0 / sqrt(128);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnFusedFloydAttentionGrad第一段接口
    ret = aclnnFusedFloydAttentionGradGetWorkspaceSize(q, k1, v1, k2, v2, dx, attenmask, softmaxMax, softmaxSum,
                                                       attentionIn, scaleValue, dq, dk1, dv1, dk2, dv2, &workspaceSize,
                                                       &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnFusedFloydAttentionGrad第二段接口
    ret = aclnnFusedFloydAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(dqShape, &dqDeviceAddr);
    // PrintOutResult(dkShape, &dkDeviceAddr);
    // PrintOutResult(dvShape, &dvDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k1);
    aclDestroyTensor(v1);
    aclDestroyTensor(k2);
    aclDestroyTensor(v2);
    aclDestroyTensor(dx);
    aclDestroyTensor(attenmask);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(attentionIn);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk1);
    aclDestroyTensor(dv1);
    aclDestroyTensor(dk2);
    aclDestroyTensor(dv2);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(k1DeviceAddr);
    aclrtFree(v1DeviceAddr);
    aclrtFree(k2DeviceAddr);
    aclrtFree(v2DeviceAddr);
    aclrtFree(dxDeviceAddr);
    aclrtFree(attenmaskDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(attentionInDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dk1DeviceAddr);
    aclrtFree(dv1DeviceAddr);
    aclrtFree(dk2DeviceAddr);
    aclrtFree(dv2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
