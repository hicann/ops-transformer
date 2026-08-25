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
 * \file test_aclnn_dense_lightning_indexer_softmax_lse_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
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
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // BSND layout: q[B,S1,N1,D] k[B,S2,N2,D] w[B,S1,N1] lse[B,N2,S1]
    int64_t B = 2;
    int64_t S1 = 128;
    int64_t S2 = 256;
    int64_t N1 = 8;
    int64_t N2 = 1;
    int64_t D = 128;

    std::vector<int64_t> qIndexShape = {B, S1, N1, D};
    std::vector<int64_t> kIndexShape = {B, S2, N2, D};
    std::vector<int64_t> weightShape = {B, S1, N1};
    std::vector<int64_t> softmaxLseShape = {B, N2, S1};
    std::vector<int64_t> metadataShape = {64};

    void *qIndexDeviceAddr = nullptr;
    void *kIndexDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;

    aclTensor *qIndex = nullptr;
    aclTensor *kIndex = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *softmaxLse = nullptr;
    aclTensor *metadata = nullptr;

    // 使用 float16 作为输入数据类型，weight 和输出 lse 固定为 float32
    std::vector<aclFloat16> qIndexHostData(B * S1 * N1 * D, aclFloatToFloat16(0.2));
    std::vector<aclFloat16> kIndexHostData(B * S2 * 1 * D, aclFloatToFloat16(0.1));
    std::vector<float> weightHostData(B * S1 * N1, 0.005f);
    std::vector<float> softmaxLseHostData(B * N2 * S1, 0.0f);

    // 前置metadata算子输出的分核负载均衡信息，shape (64,)，dtype int32
    std::vector<int32_t> metadataHostData(64, 0);

    ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // BSND layout, maskMode=0 (defaultMask), cmpRatio=1
    constexpr const char layoutQStr[] = "BSND";
    constexpr const char layoutKStr[] = "BSND";
    constexpr size_t layoutQLen = sizeof(layoutQStr);
    constexpr size_t layoutKLen = sizeof(layoutKStr);
    char layoutQ[layoutQLen];
    char layoutK[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;

    // 3. 调用 metadata 前置算子，生成分核负载均衡信息
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, B, S1, S2, N1, N2, D, layoutQ, layoutK, maskMode, cmpRatio,
        metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                         stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 4. 调用主算子
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 调用aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize第一段接口
    ret = aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(qIndex, kIndex, weight, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr, metadata, layoutQ, layoutK, maskMode,
                                                                 cmpRatio, softmaxLse, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnDenseLightningIndexerSoftmaxLseV2第二段接口
    ret = aclnnDenseLightningIndexerSoftmaxLseV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2 failed. ERROR: %d\n", ret);
              return ret);

    // 5. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(softmaxLseShape, &softmaxLseDeviceAddr);
    LOG_PRINT("pass\n");

    // 7. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(qIndex);
    aclDestroyTensor(kIndex);
    aclDestroyTensor(weight);
    aclDestroyTensor(softmaxLse);
    aclDestroyTensor(metadata);

    // 8. 释放device资源
    aclrtFree(qIndexDeviceAddr);
    aclrtFree(kIndexDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(metadataDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
