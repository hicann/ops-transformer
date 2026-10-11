/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k.h"
#include <iostream>
#include <vector>
#include <random>

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

std::vector<float> GenerateRandomFloats(int64_t count)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> result(count);
    for (auto& num : result) {
        num = dist(gen);
    }
    return result;
}
int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
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

struct TensorResources {
    void* inputDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* expertIdOutDeviceAddr = nullptr;
    void* normOutDeviceAddr = nullptr;
    aclTensor* inputTensor = nullptr;
    aclTensor* biasTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensor* expertIdOutTensor = nullptr;
    aclTensor* normOutTensor = nullptr;
};

void CleanupResources(TensorResources& resources, void* workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.inputTensor) {
        aclDestroyTensor(resources.inputTensor);
    }
    if (resources.biasTensor) {
        aclDestroyTensor(resources.biasTensor);
    }
    if (resources.outTensor) {
        aclDestroyTensor(resources.outTensor);
    }
    if (resources.expertIdOutTensor) {
        aclDestroyTensor(resources.expertIdOutTensor);
    }
    if (resources.normOutTensor) {
        aclDestroyTensor(resources.normOutTensor);
    }
    if (resources.inputDeviceAddr) {
        aclrtFree(resources.inputDeviceAddr);
    }
    if (resources.biasDeviceAddr) {
        aclrtFree(resources.biasDeviceAddr);
    }
    if (resources.outDeviceAddr) {
        aclrtFree(resources.outDeviceAddr);
    }
    if (resources.expertIdOutDeviceAddr) {
        aclrtFree(resources.expertIdOutDeviceAddr);
    }
    if (resources.normOutDeviceAddr) {
        aclrtFree(resources.normOutDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == 0)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {3, 256};
    std::vector<int64_t> biasShape = {256};
    std::vector<int64_t> outShape = {3, 8};
    std::vector<int64_t> expertIdOutShape = {3, 8};
    std::vector<int64_t> normOutShape = {3, 256};

    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<float> inputHostData = GenerateRandomFloats(GetShapeSize(inputShape));
    std::vector<float> biasHostData = GenerateRandomFloats(GetShapeSize(biasShape));
    std::vector<float> outHostData(GetShapeSize(outShape));
    std::vector<int32_t> expertIdOutHostData(GetShapeSize(expertIdOutShape));
    std::vector<float> normOutHostData(GetShapeSize(normOutShape));

    // 创建expandedPermutedRows aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &resources.inputDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.inputTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    // 创建expandedPermutedRows aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &resources.biasDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.biasTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    // 创建expertForSourceRow aclTensor
    ret =
        CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr, aclDataType::ACL_FLOAT, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    // 创建expandedSrcToDstRow aclTensor
    ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &resources.expertIdOutDeviceAddr,
                          aclDataType::ACL_INT32, &resources.expertIdOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    // 创建expandedSrcToDstRow aclTensor
    ret = CreateAclTensor(normOutHostData, normOutShape, &resources.normOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.normOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3.调用CANN算子库API，需要修改为具体的算子接口
    aclOpExecutor* executor;

    // 调用aclnnMoeGatingTopK第一段接口
    ret = aclnnMoeGatingTopKGetWorkspaceSize(resources.inputTensor, resources.biasTensor, 8, 4, 8, 1, 0, 1, false, 1, 1,
                                             resources.outTensor, resources.expertIdOutTensor, resources.normOutTensor,
                                             &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMoeGatingTopKGetWorkspaceSize failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            CleanupResources(resources, workspaceAddr, stream, deviceId);
            return ret;
        }
    }
    // 调用aclnnMoeGatingTopK第二段接口
    ret = aclnnMoeGatingTopK(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMoeGatingTopK failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4.（ 固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), resources.outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 清理资源
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
