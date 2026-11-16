/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_engram_fused_weight.cpp
 * \brief aclnnEngramFusedWeight 单算子调用样例 (两段式接口)
 */

#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_engram_fused_weight.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法, 资源初始化
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
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出: 两个 BF16 权重 [hcMult, hiddenSize] = [4, 5120], 输出 FP32 同 shape
    std::vector<int64_t> weightShape = {4, 5120};
    void* weightHiddenDeviceAddr = nullptr;
    void* weightEmbedDeviceAddr = nullptr;
    void* weightFusedDeviceAddr = nullptr;
    aclTensor* weightHidden = nullptr;
    aclTensor* weightEmbed = nullptr;
    aclTensor* weightFused = nullptr;
    int64_t weightElemNum = GetShapeSize(weightShape);
    std::vector<uint16_t> weightHiddenHostData(weightElemNum, 0x3F80); // bf16 1.0
    std::vector<uint16_t> weightEmbedHostData(weightElemNum, 0xBF00);  // bf16 -0.5
    std::vector<float> weightFusedHostData(weightElemNum, 0.0f);
    ret = CreateAclTensor(weightHiddenHostData, weightShape, &weightHiddenDeviceAddr, aclDataType::ACL_BF16,
                          &weightHidden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(weightEmbedHostData, weightShape, &weightEmbedDeviceAddr, aclDataType::ACL_BF16, &weightEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(weightFusedHostData, weightShape, &weightFusedDeviceAddr, aclDataType::ACL_FLOAT, &weightFused);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API: aclnn 两段式接口 (无属性)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnEngramFusedWeight第一段接口
    ret = aclnnEngramFusedWeightGetWorkspaceSize(weightHidden, weightEmbed, weightFused, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramFusedWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnEngramFusedWeight第二段接口
    ret = aclnnEngramFusedWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramFusedWeight failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.拷回 host 校验: 期望 1.0 * -0.5 = -0.5
    std::vector<float> result(weightElemNum, 0.0f);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(float), weightFusedDeviceAddr,
                      weightElemNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    float maxErr = 0.0f;
    for (int64_t i = 0; i < weightElemNum; i++) {
        maxErr = std::max(maxErr, std::fabs(result[i] - (-0.5f)));
    }
    LOG_PRINT("aclnnEngramFusedWeight done, max error = %f\n", maxErr);

    // 6.释放aclTensor
    aclDestroyTensor(weightHidden);
    aclDestroyTensor(weightEmbed);
    aclDestroyTensor(weightFused);

    // 7.释放device资源
    aclrtFree(weightHiddenDeviceAddr);
    aclrtFree(weightEmbedDeviceAddr);
    aclrtFree(weightFusedDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
