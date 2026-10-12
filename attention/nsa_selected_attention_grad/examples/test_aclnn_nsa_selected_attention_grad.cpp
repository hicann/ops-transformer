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
 * \file test_aclnn_nsa_selected_attention_grad.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_selected_attention_grad.h"

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
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t b = 1;
    int64_t s1 = 1;
    int64_t s2 = 1024;
    int64_t t1 = b * s1;
    int64_t t2 = b * s2;
    int64_t n1 = 1;
    int64_t n2 = 1;
    int64_t d = 192;

    int64_t sparseMode = 0;
    char inputLayout[5] = {'T', 'N', 'D', 0};
    double scaleValue = 1.0f;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int32_t headNum = n1;

    std::vector<int64_t> queryShape = {t1, n1, d};
    std::vector<int64_t> keyShape = {t2, n2, d};
    std::vector<int64_t> valueShape = {t2, n2, d};
    std::vector<int64_t> attentionOutShape = {t1, n1, d};
    std::vector<int64_t> attentionOutGradShape = {t1, n1, d};
    std::vector<int64_t> softmaxMaxShape = {t1, n1, 8};
    std::vector<int64_t> softmaxSumShape = {t1, n1, 8};
    std::vector<int64_t> topkIndicesShape = {t1, n2, selectedBlockCount};
    std::vector<int64_t> actualSeqQLenOptionalShape = {b};
    std::vector<int64_t> actualSeqKvLenOptionalShape = {b};
    std::vector<int64_t> dqOutShape = {t1, n1, d};
    std::vector<int64_t> dkOutShape = {t2, n2, d};
    std::vector<int64_t> dvOutShape = {t2, n2, d};

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* attentionOutGradDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* topkIndicesDeviceAddr = nullptr;
    void* dqOutDeviceAddr = nullptr;
    void* dkOutDeviceAddr = nullptr;
    void* dvOutDeviceAddr = nullptr;

    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* attentionOutGrad = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* topkIndices = nullptr;
    aclTensor* dqOut = nullptr;
    aclTensor* dkOut = nullptr;
    aclTensor* dvOut = nullptr;

    std::vector<aclFloat16> queryHostData(GetShapeSize(queryShape), 2);
    std::vector<aclFloat16> keyHostData(GetShapeSize(keyShape), 2);
    std::vector<aclFloat16> valueHostData(GetShapeSize(valueShape), 2);
    std::vector<aclFloat16> attentionOutHostData(GetShapeSize(attentionOutShape), 2);
    std::vector<aclFloat16> attentionOutGradHostData(GetShapeSize(attentionOutGradShape), 2);
    std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 2);
    std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 2);
    std::vector<int32_t> topkIndicesHostData(GetShapeSize(topkIndicesShape), 1);
    std::vector<aclFloat16> dqOutHostData(GetShapeSize(dqOutShape), 2);
    std::vector<aclFloat16> dkOutHostData(GetShapeSize(dkOutShape), 2);
    std::vector<aclFloat16> dvOutHostData(GetShapeSize(dvOutShape), 2);

    for (int32_t i = 0; i < topkIndicesHostData.size(); i++) {
        topkIndicesHostData[i] = i;
    }

    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建attentionOut aclTensor
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建attentionOutGrad aclTensor
    ret = CreateAclTensor(attentionOutGradHostData, attentionOutGradShape, &attentionOutGradDeviceAddr,
                          aclDataType::ACL_FLOAT16, &attentionOutGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建softmaxMax aclTensor
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建softmaxSum aclTensor
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建topkIndices aclTensor
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &topkIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int64_t tempQ[1] = {1};
    int64_t tempK[1] = {1024};
    aclIntArray* actualSeqQLenOptional = aclCreateIntArray(tempQ, static_cast<uint64_t>(1));
    aclIntArray* actualSeqKvLenOptional = aclCreateIntArray(tempK, static_cast<uint64_t>(1));
    // 创建dq aclTensor
    ret = CreateAclTensor(dqOutHostData, dqOutShape, &dqOutDeviceAddr, aclDataType::ACL_FLOAT16, &dqOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dk aclTensor
    ret = CreateAclTensor(dkOutHostData, dkOutShape, &dkOutDeviceAddr, aclDataType::ACL_FLOAT16, &dkOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dv aclTensor
    ret = CreateAclTensor(dvOutHostData, dvOutShape, &dvOutDeviceAddr, aclDataType::ACL_FLOAT16, &dvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnNsaSelectedAttentionGrad接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNsaSelectedAttentionGrad第一段接口
    ret = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum, topkIndices, actualSeqQLenOptional,
        actualSeqKvLenOptional, nullptr, scaleValue, selectedBlockSize, selectedBlockCount, headNum, inputLayout,
        sparseMode, dqOut, dkOut, dvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnNsaSelectedAttentionGrad第二段接口
    ret = aclnnNsaSelectedAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto dqSize = GetShapeSize(dqOutShape);
    std::vector<aclFloat16> dqResultData(dqSize, 0);
    ret = aclrtMemcpy(dqResultData.data(), dqResultData.size() * sizeof(dqResultData[0]), dqOutDeviceAddr,
                      dqSize * sizeof(dqResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dq from device to host failed. ERROR: %d\n", ret);
              return ret);
    auto dkSize = GetShapeSize(dkOutShape);
    std::vector<aclFloat16> dkResultData(dkSize, 0);
    ret = aclrtMemcpy(dkResultData.data(), dkResultData.size() * sizeof(dkResultData[0]), dkOutDeviceAddr,
                      dkSize * sizeof(dkResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dk from device to host failed. ERROR: %d\n", ret);
              return ret);
    auto dvSize = GetShapeSize(dvOutShape);
    std::vector<aclFloat16> dvResultData(dkSize, 0);
    ret = aclrtMemcpy(dvResultData.data(), dvResultData.size() * sizeof(dvResultData[0]), dkOutDeviceAddr,
                      dvSize * sizeof(dvResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dv from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(attentionOut);
    aclDestroyTensor(attentionOutGrad);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(topkIndices);
    aclDestroyTensor(dqOut);
    aclDestroyTensor(dkOut);
    aclDestroyTensor(dvOut);
    aclDestroyIntArray(actualSeqQLenOptional);
    aclDestroyIntArray(actualSeqKvLenOptional);
    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(attentionOutGradDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(dqOutDeviceAddr);
    aclrtFree(dkOutDeviceAddr);
    aclrtFree(dvOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
