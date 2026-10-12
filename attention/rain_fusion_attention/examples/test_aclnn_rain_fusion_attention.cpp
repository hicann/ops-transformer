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
 * \file test_aclnn_rain_fusion_attention.cpp
 * \brief RainFusionAttention 算子测试用例
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_rain_fusion_attention.h"

using namespace std;

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
    // 固定写法，AscendCL初始化
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
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", hostData.size(),
                  GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return -1);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置参数
    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;

    // 计算TND格式维度
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;   // Q块的X维度数量
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY; // KV块的Y维度数量
    // totalQBlocks = qBlockNum * numHeads (每个Q块对应一个head)
    int32_t totalQBlocks = qBlockNum * numHeads;
    int32_t maxKvBlockNum = kvBlockNum;

    // 3. 创建Query tensor (TND format: [totalQTokens, numHeads, headDim])
    void* queryDeviceAddr = nullptr;
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    aclTensor* queryTensor = nullptr;
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);

    // 4. 创建Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n"); return ret);

    // 5. 生成稀疏索引 selectIdx 和 selectNumIdx
    // selectIdx: [totalQBlocks, numHeads, maxKvBlockNum] - 三维tensor
    // selectNumIdx: [totalQBlocks, numHeads] - 二维tensor
    // 稀疏率为1，即不做稀疏，每个Q块选择所有KV块
    std::vector<int64_t> selectIdxHostData(totalQBlocks * numHeads * maxKvBlockNum, -1);
    std::vector<int64_t> selectNumIdxHostData(totalQBlocks * numHeads, 0);

    // 稀疏率为1：每个Q块选择所有KV块，直接给下标0到maxKvBlockNum-1
    for (int32_t qb = 0; qb < totalQBlocks; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            // selectNumIdx[qb, h] = maxKvBlockNum (每个Q块选择所有KV块)
            selectNumIdxHostData[qb * numHeads + h] = static_cast<int64_t>(maxKvBlockNum);

            // selectIdx[qb, h, k] = k (直接给下标，从0到maxKvBlockNum-1)
            int64_t baseIdx = static_cast<int64_t>((qb * numHeads + h) * maxKvBlockNum);
            for (int32_t k = 0; k < maxKvBlockNum; ++k) {
                selectIdxHostData[baseIdx + k] = static_cast<int64_t>(k);
            }
        }
    }

    void* selectIdxDeviceAddr = nullptr;
    void* selectNumIdxDeviceAddr = nullptr;
    std::vector<int64_t> selectIdxShape = {totalQBlocks, numHeads, maxKvBlockNum};
    std::vector<int64_t> selectNumIdxShape = {totalQBlocks, numHeads};
    aclTensor* selectIdxTensor = nullptr;
    aclTensor* selectNumIdxTensor = nullptr;
    ret = CreateAclTensor(selectIdxHostData, selectIdxShape, &selectIdxDeviceAddr, aclDataType::ACL_INT64,
                          &selectIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectIdx tensor\n"); return ret);
    ret = CreateAclTensor(selectNumIdxHostData, selectNumIdxShape, &selectNumIdxDeviceAddr, aclDataType::ACL_INT64,
                          &selectNumIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectNumIdx tensor\n"); return ret);

    // 6. 创建输出tensor
    void* outputDeviceAddr = nullptr;
    std::vector<int64_t> outputShape = {totalQTokens, numHeads, headDim};
    int64_t outputElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> outputHostData(outputElementCount, 0.0f);
    aclTensor* outputTensor = nullptr;
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create output tensor\n"); return ret);

    // 7. 创建blockShape数组
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray* blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n"); return -1);

    // 8. 创建actualSeqLengths和actualSeqLengthsKv (必需参数)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));

    void* actualSeqLengthsDevice = nullptr;
    void* actualSeqLengthsKvDevice = nullptr;
    size_t seqLengthsSize = batch * sizeof(int64_t);

    ret = aclrtMalloc(&actualSeqLengthsDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n"); return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n");
              aclrtFree(actualSeqLengthsDevice); return ret);

    ret = aclrtMemcpy(actualSeqLengthsDevice, seqLengthsSize, actualSeqLengthsHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n");
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDevice, seqLengthsSize, actualSeqLengthsKvHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n");
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);

    // aclCreateIntArray 期望的是 host 侧的数据指针，而不是 device 侧的数据
    aclIntArray* actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    aclIntArray* actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
              if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice); return -1);

    // 9. 准备字符串参数（确保缓冲区大小足够，包含null terminator）
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    // 10. 计算scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // 11. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnRainFusionAttentionGetWorkspaceSize(queryTensor,        // query
                                                   keyTensor,          // key
                                                   valueTensor,        // value
                                                   selectIdxTensor,    // selectIdx
                                                   selectNumIdxTensor, // selectNumIdx
                                                   blockShape,         // blockShape
                                                   nullptr,            // attenMaskOptional
                                                   actualSeqLengths,   // actualSeqLengthsOptional
                                                   actualSeqLengthsKv, // actualSeqLengthsKvOptional
                                                   nullptr,            // blockTableOptional
                                                   qLayoutBuffer,      // qInputLayout
                                                   kvLayoutBuffer,     // kvInputLayout
                                                   numKvHeads,         // numKeyValueHeads
                                                   0,                  // maskType
                                                   scaleValue,         // scaleValue
                                                   0,                  // innerPrecise (1=fp16 softmax)
                                                   128,                // blockSize
                                                   outputTensor,       // attentionOut
                                                   nullptr,            // softmaxLseOptional
                                                   &workspaceSize,     // workspaceSize (out)
                                                   &executor);         // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRainFusionAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);

    // 12. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 12. 调用第二段接口
    ret = aclnnRainFusionAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRainFusionAttention failed. ERROR: %d\n", ret); return ret);

    // 13. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 14. 获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t outputSize = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(outputSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), outputDeviceAddr,
                      outputSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 15. 打印部分结果
    uint64_t printNum = 10;
    LOG_PRINT("Output results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    // 16. 释放资源
    if (workspaceAddr)
        aclrtFree(workspaceAddr);
    if (queryDeviceAddr)
        aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr)
        aclrtFree(keyDeviceAddr);
    if (valueDeviceAddr)
        aclrtFree(valueDeviceAddr);
    if (outputDeviceAddr)
        aclrtFree(outputDeviceAddr);
    if (selectIdxDeviceAddr)
        aclrtFree(selectIdxDeviceAddr);
    if (selectNumIdxDeviceAddr)
        aclrtFree(selectNumIdxDeviceAddr);
    if (actualSeqLengthsDevice)
        aclrtFree(actualSeqLengthsDevice);
    if (actualSeqLengthsKvDevice)
        aclrtFree(actualSeqLengthsKvDevice);

    if (queryTensor)
        aclDestroyTensor(queryTensor);
    if (keyTensor)
        aclDestroyTensor(keyTensor);
    if (valueTensor)
        aclDestroyTensor(valueTensor);
    if (outputTensor)
        aclDestroyTensor(outputTensor);
    if (selectIdxTensor)
        aclDestroyTensor(selectIdxTensor);
    if (selectNumIdxTensor)
        aclDestroyTensor(selectNumIdxTensor);
    if (blockShape)
        aclDestroyIntArray(blockShape);
    if (actualSeqLengths)
        aclDestroyIntArray(actualSeqLengths);
    if (actualSeqLengthsKv)
        aclDestroyIntArray(actualSeqLengthsKv);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
