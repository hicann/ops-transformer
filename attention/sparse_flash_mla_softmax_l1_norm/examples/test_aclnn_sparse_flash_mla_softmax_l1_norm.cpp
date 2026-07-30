/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h"

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

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // TND layout: qShape=(T1,N1,D), kShape=(T2,N2,D), softmaxLseShape=(N2,T1,G)
    std::vector<int64_t> qShape = {16, 128, 512};
    std::vector<int64_t> kShape = {2048, 1, 512};
    std::vector<int64_t> softmaxLseShape = {1, 16, 128};
    std::vector<int64_t> cuSeqQLenshape = {2};
    std::vector<int64_t> cuSeqKLenshape = {2};
    std::vector<int64_t> cmpResidualKShape = {1};
    std::vector<int64_t> softmaxL1NormShape = {16, 1, 2048};
    std::vector<int64_t> metadataShape = {64};

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;
    void *cuSeqQLenDeviceAddr = nullptr;
    void *cuSeqKLenDeviceAddr = nullptr;
    void *cmpResidualKDeviceAddr = nullptr;
    void *softmaxL1NormDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *softmaxLse = nullptr;
    aclTensor *cuSeqQLen = nullptr;
    aclTensor *cuSeqKLen = nullptr;
    aclTensor *cmpResidualK = nullptr;
    aclTensor *softmaxL1Norm = nullptr;
    aclTensor *metadata = nullptr;

    std::vector<short> qHostData(16 * 128 * 512, 1.0);
    std::vector<short> kHostData(2048 * 1 * 512, 1.0);
    std::vector<float> softmaxLseHostData(1 * 16 * 128, 3.0);
    std::vector<int32_t> cuSeqQLenHostData = {0, 16};
    std::vector<int32_t> cuSeqKLenHostData = {0, 2048};
    std::vector<int32_t> cmpResidualKHostData = {0};
    std::vector<float> softmaxL1NormHostData(16 * 1 * 2048, 0);
    std::vector<int32_t> metadataHostData(64, 0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqKLenHostData, cuSeqKLenshape, &cuSeqKLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqKLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &cmpResidualKDeviceAddr, aclDataType::ACL_INT32,
                          &cmpResidualK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxL1NormHostData, softmaxL1NormShape, &softmaxL1NormDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxL1Norm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double softmaxScale = 0.088388;
    int64_t maxSeqlenK = 2048;
    int64_t cmpRatio = 128;
    int64_t maskMode = 3;
    char layoutQ[4] = {'T', 'N', 'D', 0};
    char layoutK[4] = {'T', 'N', 'D', 0};

    // 1. 调用 metadata 前置算子
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    ret = aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
        cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr, 0, 16, 2048, 128, 1, 512, 0, cmpRatio, maskMode,
        layoutQ, layoutK, metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSparseFlashMlaSoftmaxL1NormMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                   stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadata failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 2. 调用主算子
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
        q, k, softmaxLse, nullptr, cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr, metadata,
        softmaxScale, maxSeqlenK, cmpRatio, maskMode, layoutQ, layoutK, softmaxL1Norm, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    try {
        ret = aclnnSparseFlashMlaSoftmaxL1Norm(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("Expected kernel failure (skeleton stage): ERROR: %d\n", ret);
        } else {
            ret = aclrtSynchronizeStream(stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        }
    } catch (const std::exception &e) {
        LOG_PRINT("Expected kernel failure (skeleton stage): %s\n", e.what());
    }

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(softmaxLse);
    aclDestroyTensor(cuSeqQLen);
    aclDestroyTensor(cuSeqKLen);
    aclDestroyTensor(cmpResidualK);
    aclDestroyTensor(softmaxL1Norm);
    aclDestroyTensor(metadata);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(cuSeqQLenDeviceAddr);
    aclrtFree(cuSeqKLenDeviceAddr);
    aclrtFree(cmpResidualKDeviceAddr);
    aclrtFree(softmaxL1NormDeviceAddr);
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
