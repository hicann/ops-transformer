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
#include "aclnnop/aclnn_moe_token_permute_with_ep_grad.h"
#include <iostream>
#include <vector>

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
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray)
{
    auto size = GetShapeSize(hostData) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call aclCreateIntArray to create an aclIntArray.
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
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

    int64_t num_topk = 2;
    std::vector<float> permuted_token_output_grad_Data = {2, 2, 1, 1, 3, 3, 2, 2};
    std::vector<float> permuted_prob_output_grad_Data = {0.2, 0.5, 0.4, 0.4};
    std::vector<int64_t> permuted_token_output_grad_Shape = {4, 2};
    std::vector<int64_t> permuted_prob_output_grad_Shape = {4};
    void* permuted_token_output_grad_Addr = nullptr;
    void* permuted_prob_output_grad_Addr = nullptr;
    aclTensor* permuted_token_output_grad = nullptr;
    aclTensor* permuted_prob_output_grad = nullptr;

    ret = CreateAclTensor(permuted_token_output_grad_Data, permuted_token_output_grad_Shape,
                          &permuted_token_output_grad_Addr, aclDataType::ACL_BF16, &permuted_token_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(permuted_prob_output_grad_Data, permuted_prob_output_grad_Shape,
                          &permuted_prob_output_grad_Addr, aclDataType::ACL_BF16, &permuted_prob_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
    std::vector<int64_t> sortedIndicesShape = {6};
    void* sortedIndicesAddr = nullptr;
    aclTensor* sortedIndices = nullptr;

    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* rangeDeviceAddr = nullptr;
    aclIntArray* range = nullptr;
    std::vector<int64_t> rangeHostData = {1, 5};
    ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> tokenOutData = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> tokenOutShape = {3, 2};
    void* tokenOutAddr = nullptr;
    aclTensor* tokenOut = nullptr;

    ret = CreateAclTensor(tokenOutData, tokenOutShape, &tokenOutAddr, aclDataType::ACL_BF16, &tokenOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> probOutData = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> probOutShape = {3, 2};
    void* probOutAddr = nullptr;
    aclTensor* probOut = nullptr;

    ret = CreateAclTensor(probOutData, probOutShape, &probOutAddr, aclDataType::ACL_BF16, &probOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeTokenPermuteWithEpGrad第一段接口
    ret = aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(permuted_token_output_grad, sortedIndices,
                                                         permuted_prob_output_grad, num_topk, range, false, tokenOut,
                                                         probOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMoeTokenPermuteWithEpGrad第二段接口
    ret = aclnnMoeTokenPermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(tokenOutShape, &tokenOutAddr);
    PrintOutResult(probOutShape, &probOutAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permuted_token_output_grad);
    aclDestroyTensor(permuted_prob_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(tokenOut);
    aclDestroyTensor(probOut);

    // 7. 释放device资源
    aclrtFree(permuted_token_output_grad_Addr);
    aclrtFree(permuted_prob_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
    aclrtFree(tokenOutAddr);
    aclrtFree(probOutAddr);
    aclrtFree(rangeDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
