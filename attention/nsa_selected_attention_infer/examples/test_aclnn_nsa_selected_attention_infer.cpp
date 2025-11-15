/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nsa_selected_attention_infer.cpp
 * \brief
 */

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_select_attention_infer.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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

int main(int argc, char **argv)
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_selected_attention_infer 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行

    int64_t batch = 1;
    int sequenceLengthK = 48;
    aclIntArray * actualCmpKvSeqLen = nullptr;
    aclIntArray * actualCmpQSeqLen = nullptr;
    // 创建actualCmpKvSeqLen aclIntArray
    std::vector<int64_t> actualCmpKvSeqLenVector(batch, sequenceLengthK);
    actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    // 创建actualCmpQSeqLen aclIntArray
    int64_t s1 = 1;
    std::vector<int64_t> actualCmpQSeqLenVector(batch, s1);
    actualCmpQSeqLen = aclCreateIntArray(actualCmpQSeqLenVector.data(), actualCmpQSeqLenVector.size());
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 1;
    
    int64_t n2 = 1;
    int64_t blockSize = 64;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 1;
    int64_t blockTableLength = 1;
    int64_t numBlocks = batch * blockTableLength;
    std::vector<int64_t> queryShape = {batch, s1, n2 * g, d1};
    std::vector<int64_t> keyShape = {numBlocks, blockSize, n2,d1};
    std::vector<int64_t> valueShape = {numBlocks, blockSize, n2,d2};
    std::vector<int64_t> topkIndicesShape = {batch, s1, n2, selectBlockCount};
    std::vector<int64_t> blockTableOptionalShape = {batch, blockTableLength};
    std::vector<int64_t> outputShape = {batch, s1, n2 * g, d2};

    long long queryShapeSize = GetShapeSize(queryShape);
    long long keyShapeSize = GetShapeSize(keyShape);
    long long valueShapeSize = GetShapeSize(valueShape);
    long long blockTableOptionalShapeSize = GetShapeSize(blockTableOptionalShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long topkIndicesShapeSize = GetShapeSize(topkIndicesShape);

    std::vector<int16_t> queryHostData(queryShapeSize, 1);
    std::vector<int16_t> keyHostData(keyShapeSize, 1);
    std::vector<int16_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> blockTableOptionalHostData(blockTableOptionalShapeSize, 0);
    std::vector<int16_t> outputHostData(outputShapeSize, 1);
    
    std::vector<int32_t> topkIndicesHostData;
    for (int b = 0; b < batch; ++b) {
       for (int s = 0; s < s1; ++s) {
        for (int h = 0; h < n2; ++h) {
            for (int k = 0; k < selectBlockCount; ++k) {
                if (k == 0) {
                    topkIndicesHostData.push_back(k);
                } else {
                    topkIndicesHostData.push_back(-1);
                }
            }
        }
       }
    }
    // attr
    double scaleValue = 1.0;
    int64_t sparseMod = 0;
    int64_t numHeads= static_cast<int64_t>(n2 * g);
    std::string sLayerOut = "BSND";
    char layOut[sLayerOut.length()];
    std::strcpy(layOut, sLayerOut.c_str());

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockTableOptionalDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *topkIndicesDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockTableOptionalTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclTensor *topkIndicesTensor = nullptr;
    
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建blockTableOptional aclTensor
    ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建topkIndices aclTensor
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnNsaSelectedAttention第一段接口
    ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, topkIndicesTensor, nullptr,
                blockTableOptionalTensor, actualCmpQSeqLen, actualCmpKvSeqLen, layOut,
                numHeads, n2, selectBlockSize, selectBlockCount, blockSize,
                scaleValue, sparseMod, outputTensor,
                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnNsaSelectedAttention第二段接口
    ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnn execute success : %d\n", ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
    uint64_t printNum = 10;
    for (int64_t i = 0; i < printNum; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(topkIndicesTensor);
    aclDestroyTensor(blockTableOptionalTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(blockTableOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}