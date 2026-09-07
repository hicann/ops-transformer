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
 * \file test_aclnn_swin_attention_score_quant.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_attention_score_quant.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t B = 1288;
    int64_t N = 3;
    int64_t S = 49;
    int64_t H = 32;
    std::vector<int64_t> qkvShape = {B, N, S, H};
    std::vector<int64_t> sShape = {1, S};
    std::vector<int64_t> hShape = {1, H};
    std::vector<int64_t> mask1Shape = {1, N, S, S};
    std::vector<int64_t> attentionScoreShape = {B, N, S, H};
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *scaleQuantDeviceAddr = nullptr;
    void *scaleDequant1DeviceAddr = nullptr;
    void *scaleDequant2DeviceAddr = nullptr;
    void *biasQuantDeviceAddr = nullptr;
    void *biasDequant1DeviceAddr = nullptr;
    void *biasDequant2DeviceAddr = nullptr;
    void *paddingMask1DeviceAddr = nullptr;
    void *attentionScoreDeviceAddr = nullptr;
    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *scaleQuant = nullptr;
    aclTensor *scaleDequant1 = nullptr;
    aclTensor *scaleDequant2 = nullptr;
    aclTensor *biasQuantOptional = nullptr;
    aclTensor *biasDequant1Optional = nullptr;
    aclTensor *biasDequant2Optional = nullptr;
    aclTensor *paddingMask1Optional = nullptr;
    aclTensor *paddingMask2Optional = nullptr;
    aclTensor *attentionScore = nullptr;
    std::vector<int8_t> queryHostData(B * N * S * H, 1);
    std::vector<int8_t> keyHostData(B * N * S * H, 1);
    std::vector<int8_t> valueHostData(B * N * S * H, 1);
    std::vector<uint16_t> scaleQuantHostData(S, 1);
    std::vector<uint64_t> scaleDequant1HostData(S, 1);
    std::vector<uint64_t> scaleDequant2HostData(H, 1);
    std::vector<uint16_t> biasQuantHostData(S, 1);
    std::vector<int32_t> biasDequant1HostData(S, 1);
    std::vector<int32_t> biasDequant2HostData(H, 1);
    std::vector<uint16_t> paddingMask1HostData(1 * N * S * S, 1);
    std::vector<uint16_t> attentionScoreHostData(B * N * S * H, 0);
    // 创建input aclTensor
    ret = CreateAclTensor(queryHostData, qkvShape, &queryDeviceAddr, aclDataType::ACL_INT8, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkvShape, &keyDeviceAddr, aclDataType::ACL_INT8, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, qkvShape, &valueDeviceAddr, aclDataType::ACL_INT8, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleQuantHostData, sShape, &scaleQuantDeviceAddr, aclDataType::ACL_FLOAT16, &scaleQuant);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant1HostData, sShape, &scaleDequant1DeviceAddr, aclDataType::ACL_UINT64,
                          &scaleDequant1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant2HostData, hShape, &scaleDequant2DeviceAddr, aclDataType::ACL_UINT64,
                          &scaleDequant2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(biasQuantHostData, sShape, &biasQuantDeviceAddr, aclDataType::ACL_FLOAT16, &biasQuantOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant1HostData, sShape, &biasDequant1DeviceAddr, aclDataType::ACL_INT32,
                          &biasDequant1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant2HostData, hShape, &biasDequant2DeviceAddr, aclDataType::ACL_INT32,
                          &biasDequant2Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(paddingMask1HostData, mask1Shape, &paddingMask1DeviceAddr, aclDataType::ACL_FLOAT16,
                          &paddingMask1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(attentionScoreHostData, attentionScoreShape, &attentionScoreDeviceAddr,
                          aclDataType::ACL_FLOAT16, &attentionScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnn接口调用示例
    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnn第一段接口
    ret = aclnnSwinAttentionScoreQuantGetWorkspaceSize(query, key, value, scaleQuant, scaleDequant1, scaleDequant2,
                                                       biasQuantOptional, biasDequant1Optional, biasDequant2Optional,
                                                       paddingMask1Optional, paddingMask2Optional, false, false, false,
                                                       -1, attentionScore, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinAttentionScoreQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnn第二段接口
    ret = aclnnSwinAttentionScoreQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinAttentionScoreQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(attentionScoreShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), attentionScoreDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(scaleQuant);
    aclDestroyTensor(scaleDequant1);
    aclDestroyTensor(scaleDequant2);
    aclDestroyTensor(biasQuantOptional);
    aclDestroyTensor(biasDequant1Optional);
    aclDestroyTensor(biasDequant2Optional);
    aclDestroyTensor(paddingMask1Optional);
    aclDestroyTensor(attentionScore);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(scaleQuantDeviceAddr);
    aclrtFree(scaleDequant1DeviceAddr);
    aclrtFree(scaleDequant2DeviceAddr);
    aclrtFree(biasQuantDeviceAddr);
    aclrtFree(biasDequant1DeviceAddr);
    aclrtFree(biasDequant2DeviceAddr);
    aclrtFree(paddingMask1DeviceAddr);
    aclrtFree(attentionScoreDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
