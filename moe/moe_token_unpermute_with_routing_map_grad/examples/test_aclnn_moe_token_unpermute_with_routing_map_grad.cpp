/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map_grad.h"
#include <iostream>

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
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
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
    bool paddedMode = false;
    int32_t tokenNum = 1;
    int32_t hiddenSize = 2;
    int32_t expertNum = 2;
    int32_t topK = 2;
    int32_t outTokenNum = tokenNum * topK;
    std::vector<int64_t> permutedTokensShape = {outTokenNum, hiddenSize};
    std::vector<int64_t> unpermutedTokensGradShape = {tokenNum, hiddenSize};
    std::vector<int64_t> probsShape = {tokenNum, expertNum};
    std::vector<int64_t> outIndexShape = {outTokenNum};
    std::vector<int64_t> permuteTokenIdShape = {outTokenNum};
    std::vector<int64_t> routingMapShape = {tokenNum, expertNum};
    std::vector<int64_t> permutedTokensGradShape = {outTokenNum, hiddenSize};
    std::vector<int64_t> probsGradShape = {tokenNum, expertNum};
    void* permutedTokensDeviceAddr = nullptr;
    void* unpermutedTokensGradDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* routingMapDeviceAddr = nullptr;
    void* permutedTokensGradDeviceAddr = nullptr;
    void* probsGradDeviceAddr = nullptr;

    aclTensor* permutedTokens = nullptr;
    aclTensor* unpermutedTokensGrad = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* routingMap = nullptr;
    aclTensor* permutedTokensGrad = nullptr;
    aclTensor* probsGrad = nullptr;

    std::vector<float> permutedTokensHostData = {1, 1, 1, 1};
    std::vector<float> unpermutedTokensGradHostData = {1, 1};
    std::vector<float> probsHostData = {1, 1};
    std::vector<int> outIndexHostData = {0, 1};
    std::vector<int> permuteTokenIdHostData = {0, 0};
    std::vector<int8_t> routingMapHostData = {1, 1};
    std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0};
    std::vector<float> probsGradHostData = {0, 0};

    ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                          aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr,
                          aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(routingMapHostData, routingMapShape, &routingMapDeviceAddr, aclDataType::ACL_BOOL, &routingMap);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr,
                          aclDataType::ACL_FLOAT, &permutedTokensGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT, &probsGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第一段接口
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(
        unpermutedTokensGrad, outIndex, permuteTokenId, routingMap, permutedTokens, probs, paddedMode, nullptr,
        permutedTokensGrad, probsGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第二段接口
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("permutedTokensGrad \n");
    PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
    LOG_PRINT("probsGrad \n");
    PrintOutResult(probsGradShape, &probsGradDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(unpermutedTokensGrad);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(routingMap);
    aclDestroyTensor(probs);
    aclDestroyTensor(permutedTokensGrad);
    aclDestroyTensor(probsGrad);

    // 7. 释放device资源
    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(unpermutedTokensGradDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(routingMapDeviceAddr);
    aclrtFree(permutedTokensGradDeviceAddr);
    aclrtFree(probsGradDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
