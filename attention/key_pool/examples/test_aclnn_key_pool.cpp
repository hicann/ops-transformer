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
 * \file test_aclnn_key_pool.cpp
 * \brief KeyPool aclnn invocation example for A2/A3.
 *
 * The example uses BSH input, BF16 projection/output tensors, FP32 state cache,
 * and enables the optional LayerNorm inputs. RoPE remains disabled by passing
 * null cos/sin tensors, as required by the current operator implementation.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_key_pool.h"

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

namespace {

using BFloat16 = uint16_t;

BFloat16 FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<BFloat16>(bits >> 16);
}

float BFloat16ToFloat(BFloat16 value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续selfOrResult的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void PrintBFloat16Tensor(const char *name, const std::vector<int64_t> &shape, void *deviceAddr)
{
    const auto elementCount = GetShapeSize(shape);
    std::vector<BFloat16> hostData(elementCount);
    const auto ret = aclrtMemcpy(hostData.data(), elementCount * sizeof(BFloat16), deviceAddr,
                                 elementCount * sizeof(BFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);
    const auto printCount = std::min<int64_t>(elementCount, 8);
    for (int64_t i = 0; i < printCount; ++i) {
        LOG_PRINT("%s[%ld] = %f\n", name, i, BFloat16ToFloat(hostData[i]));
    }
}

} // namespace

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API
    // 根据自己的实际device填写deviceId
    constexpr int32_t deviceId = 0;
    constexpr int64_t batchSize = 1;
    constexpr int64_t seqLength = 8;
    constexpr int64_t hiddenSize = 4096;
    constexpr int64_t headDim = 128;
    constexpr int64_t cmpRatio = 4;
    constexpr int64_t blockSize = 4;
    constexpr int64_t maxBlockNumPerBatch = (seqLength + blockSize - 1) / blockSize;
    constexpr int64_t blockNum = batchSize * maxBlockNumPerBatch + 1; // block 0 is reserved.
    constexpr int64_t pooledSeqLength = (maxBlockNumPerBatch * blockSize + cmpRatio - 1) / cmpRatio;
    constexpr double normEps = 1e-6;
    constexpr int64_t rotaryMode = 1;
    constexpr int64_t stateCacheStrideDim0 = blockSize * 2 * headDim;

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    const std::vector<int64_t> hiddenStatesShape = {batchSize, seqLength, hiddenSize};
    const std::vector<int64_t> wkShape = {headDim, hiddenSize};
    const std::vector<int64_t> gateWeightShape = {headDim, hiddenSize};
    const std::vector<int64_t> apeShape = {cmpRatio, headDim};
    const std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * headDim};
    const std::vector<int64_t> cacheBlockTableShape = {batchSize, maxBlockNumPerBatch};
    const std::vector<int64_t> startPosShape = {batchSize};
    const std::vector<int64_t> normShape = {headDim};
    const std::vector<int64_t> pooledKeyShape = {batchSize, pooledSeqLength, headDim};

    const auto hiddenStatesSize = GetShapeSize(hiddenStatesShape);
    const auto wkSize = GetShapeSize(wkShape);
    const auto gateWeightSize = GetShapeSize(gateWeightShape);
    const auto apeSize = GetShapeSize(apeShape);
    const auto stateCacheSize = GetShapeSize(stateCacheShape);
    const auto pooledKeySize = GetShapeSize(pooledKeyShape);

    std::vector<BFloat16> hiddenStatesHost(hiddenStatesSize);
    std::vector<BFloat16> wkHost(wkSize);
    std::vector<BFloat16> gateWeightHost(gateWeightSize);
    std::vector<float> apeHost(apeSize);
    std::vector<float> stateCacheHost(stateCacheSize, 0.0f);
    std::vector<int32_t> cacheBlockTableHost = {1, 2};
    std::vector<int32_t> startPosHost = {0};
    std::vector<float> normWeightHost(headDim, 1.0f);
    std::vector<float> normBiasHost(headDim, 0.0f);
    std::vector<BFloat16> pooledKeyHost(pooledKeySize, FloatToBFloat16(0.0f));

    for (int64_t i = 0; i < hiddenStatesSize; ++i) {
        hiddenStatesHost[i] = FloatToBFloat16(0.01f * std::sin(static_cast<float>(i % 97)));
    }
    for (int64_t i = 0; i < wkSize; ++i) {
        wkHost[i] = FloatToBFloat16(0.02f * std::cos(static_cast<float>(i % 53)));
        gateWeightHost[i] = FloatToBFloat16(0.02f * std::sin(static_cast<float>(i % 47)));
    }
    for (int64_t i = 0; i < apeSize; ++i) {
        apeHost[i] = 0.01f * static_cast<float>((i % cmpRatio) - 1);
    }

    void *hiddenStatesDeviceAddr = nullptr;
    void *wkDeviceAddr = nullptr;
    void *gateWeightDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *cacheBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *normWeightDeviceAddr = nullptr;
    void *normBiasDeviceAddr = nullptr;
    void *pooledKeyDeviceAddr = nullptr;

    aclTensor *hiddenStates = nullptr;
    aclTensor *wk = nullptr;
    aclTensor *gateWeight = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *cacheBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *normWeight = nullptr;
    aclTensor *normBias = nullptr;
    aclTensor *pooledKeyOut = nullptr;

    ret = CreateAclTensor(hiddenStatesHost, hiddenStatesShape, &hiddenStatesDeviceAddr, ACL_BF16, &hiddenStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkHost, wkShape, &wkDeviceAddr, ACL_BF16, &wk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gateWeightHost, gateWeightShape, &gateWeightDeviceAddr, ACL_BF16, &gateWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHost, apeShape, &apeDeviceAddr, ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHost, stateCacheShape, &stateCacheDeviceAddr, ACL_FLOAT, &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cacheBlockTableHost, cacheBlockTableShape, &cacheBlockTableDeviceAddr, ACL_INT32,
                          &cacheBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHost, startPosShape, &startPosDeviceAddr, ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightHost, normShape, &normWeightDeviceAddr, ACL_FLOAT, &normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normBiasHost, normShape, &normBiasDeviceAddr, ACL_FLOAT, &normBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pooledKeyHost, pooledKeyShape, &pooledKeyDeviceAddr, ACL_BF16, &pooledKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // 调用aclnnKeyPool第一段接口
    ret = aclnnKeyPoolGetWorkspaceSize(hiddenStates, wk, gateWeight, ape, stateCacheRef, cacheBlockTable, startPos,
                                       normWeight, normBias, nullptr, nullptr, nullptr, nullptr, cmpRatio, normEps,
                                       rotaryMode, stateCacheStrideDim0, pooledKeyOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKeyPoolGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnKeyPool第二段接口
    ret = aclnnKeyPool(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKeyPool failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("KeyPool execution succeeded. pooled_key shape=[%ld,%ld,%ld]\n", batchSize, pooledSeqLength, headDim);
    PrintBFloat16Tensor("pooled_key", pooledKeyShape, pooledKeyDeviceAddr);

    aclDestroyTensor(hiddenStates);
    aclDestroyTensor(wk);
    aclDestroyTensor(gateWeight);
    aclDestroyTensor(ape);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(cacheBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(normWeight);
    aclDestroyTensor(normBias);
    aclDestroyTensor(pooledKeyOut);

    aclrtFree(hiddenStatesDeviceAddr);
    aclrtFree(wkDeviceAddr);
    aclrtFree(gateWeightDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(cacheBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(normWeightDeviceAddr);
    aclrtFree(normBiasDeviceAddr);
    aclrtFree(pooledKeyDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
