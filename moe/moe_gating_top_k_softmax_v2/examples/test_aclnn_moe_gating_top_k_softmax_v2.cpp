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
#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {3, 4};
    std::vector<int64_t> outShape = {3, 2};
    std::vector<int64_t> expertIdOutShape = {3, 2};
    std::vector<int64_t> softmaxResultOutOptionalShape = {3, 4};

    void* inputAddr = nullptr;
    void* outAddr = nullptr;
    void* expertIdOutAddr = nullptr;
    void* softmaxResultOutOptionalAddr = nullptr;

    aclTensor* input = nullptr;
    aclTensor* out = nullptr;
    aclTensor* expertIdOut = nullptr;
    aclTensor* softmaxResultOutOptional = nullptr;

    std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
    std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
    std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
    std::vector<int32_t> softmaxResultOutOptionalHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // 创建expandedPermutedRows aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建expertForSourceRow aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建expandedSrcToDstRow aclTensor
    ret =
        CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建softmaxResultOutOptional aclTensor
    ret = CreateAclTensor(softmaxResultOutOptionalHostData, softmaxResultOutOptionalShape,
                          &softmaxResultOutOptionalAddr, aclDataType::ACL_FLOAT, &softmaxResultOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的算子接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeGatingTopKSoftmaxV2第一段接口
    ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(input, nullptr, 2, 0, true, out, expertIdOut,
                                                      softmaxResultOutOptional, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeGatingTopKSoftmaxV2第二段接口
    ret = aclnnMoeGatingTopKSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

    // 4.（ 固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclDestroyTensor(expertIdOut);
    aclDestroyTensor(softmaxResultOutOptional);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputAddr);
    aclrtFree(outAddr);
    aclrtFree(expertIdOutAddr);
    aclrtFree(softmaxResultOutOptionalAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
