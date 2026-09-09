/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_lightning_indexer_grad.cpp
 * \brief
 */
// testci
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_grad.h"

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

template <typename T>
void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), *deviceAddr, size * sizeof(T),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
              aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtDestroyContext(context);
              aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *dy, aclTensor *sparseIndices, aclTensor *weights,
                  aclTensor *dQuery, aclTensor *dKey, aclTensor *dWeights, void *qDeviceAddr, void *kDeviceAddr,
                  void *dyDeviceAddr, void *sparseIndicesDeviceAddr, void *weightsDeviceAddr, void *dQueryAddr,
                  void *dKeyAddr, void *dWeightsAddr, uint64_t workspaceSize, void *workspaceAddr, int32_t deviceId,
                  aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (dy != nullptr) {
        aclDestroyTensor(dy);
    }
    if (sparseIndices != nullptr) {
        aclDestroyTensor(sparseIndices);
    }
    if (weights != nullptr) {
        aclDestroyTensor(weights);
    }
    if (dQuery != nullptr) {
        aclDestroyTensor(dQuery);
    }
    if (dKey != nullptr) {
        aclDestroyTensor(dKey);
    }
    if (dWeights != nullptr) {
        aclDestroyTensor(dWeights);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (dyDeviceAddr != nullptr) {
        aclrtFree(dyDeviceAddr);
    }
    if (sparseIndicesDeviceAddr != nullptr) {
        aclrtFree(sparseIndicesDeviceAddr);
    }
    if (weightsDeviceAddr != nullptr) {
        aclrtFree(weightsDeviceAddr);
    }
    if (dQueryAddr != nullptr) {
        aclrtFree(dQueryAddr);
    }
    if (dKeyAddr != nullptr) {
        aclrtFree(dKeyAddr);
    }
    if (dWeightsAddr != nullptr) {
        aclrtFree(dWeightsAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
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
    // query、key、dy、sparseIndices、weights对应的shape值，并重新gen data，再执行
    int64_t batch = 2;
    int64_t s1 = 3;
    int64_t s2 = 2048;
    int64_t d = 128;
    int64_t g = 64;
    int64_t n2 = 1;
    int64_t topK = 2048;

    std::vector<int64_t> qShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> kShape = {batch, s2, n2, d};
    std::vector<int64_t> dyShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> sparseIndicesShape = {batch, s1, topK};
    std::vector<int64_t> weightsShape = {batch, s1, n2 * g};
    std::vector<int64_t> dQueryShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> dKeyShape = {batch, s2, n2, d};
    std::vector<int64_t> dWeightsShape = {batch, s1, n2 * g};

    int64_t headNum = 64;
    int64_t sparseMode = 3;
    char layoutStr[] = "BSND";
    bool deteminstic = true;
    int64_t preToken = 65536;
    int64_t nextToken = 65536;

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *dyDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *dQueryAddr = nullptr;
    void *dKeyAddr = nullptr;
    void *dWeightsAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *dy = nullptr;
    aclTensor *sparseIndices = nullptr;
    aclTensor *weights = nullptr;
    aclTensor *dQuery = nullptr;
    aclTensor *dKey = nullptr;
    aclTensor *dWeights = nullptr;

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1.0);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1.0);
    std::vector<aclFloat16> dyHostData(GetShapeSize(dyShape), 1.0);
    std::vector<int32_t> sparseIndicesHostData(GetShapeSize(sparseIndicesShape), 1);
    std::vector<aclFloat16> weightsHostData(GetShapeSize(weightsShape), 1.0);
    std::vector<aclFloat16> dQueryHostData(GetShapeSize(dQueryShape), 1.0);
    std::vector<aclFloat16> dKeyHostData(GetShapeSize(dKeyShape), 1.0);
    std::vector<aclFloat16> dWeightsHostData(GetShapeSize(dWeightsShape), 1.0);

    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // 创建acl Tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT16, &dy);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT16, &weights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(dQueryHostData, dQueryShape, &dQueryAddr, aclDataType::ACL_FLOAT16, &dQuery);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dKeyHostData, dKeyShape, &dKeyAddr, aclDataType::ACL_FLOAT16, &dKey);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dWeightsHostData, dWeightsShape, &dWeightsAddr, aclDataType::ACL_FLOAT16, &dWeights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnLightningIndexerGrad第一段接口
    ret = aclnnLightningIndexerGradGetWorkspaceSize(q, k, dy, sparseIndices, weights, nullptr, nullptr, headNum,
                                                    layoutStr, sparseMode, preToken, nextToken, deteminstic, dQuery,
                                                    dKey, dWeights, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                               dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr,
                               dWeightsAddr, workspaceSize, workspaceAddr, deviceId, &context, &stream);
                  return ret);
    }

    // 调用aclnnLightningIndexerGrad第二段接口
    ret = aclnnLightningIndexerGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGrad failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr,
                           dyDeviceAddr, sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr,
                           workspaceSize, workspaceAddr, deviceId, &context, &stream);
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult<aclFloat16>(dQueryShape, &dQueryAddr);
    PrintOutResult<aclFloat16>(dKeyShape, &dKeyAddr);
    PrintOutResult<aclFloat16>(dWeightsShape, &dWeightsAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                 sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize,
                 workspaceAddr, deviceId, &context, &stream);

    return 0;
}
