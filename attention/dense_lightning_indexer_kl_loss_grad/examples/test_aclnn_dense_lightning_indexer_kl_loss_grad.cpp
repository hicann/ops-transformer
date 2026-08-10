/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_dense_lightning_indexer_kl_loss_grad.cpp
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h"

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < std::min<int64_t>(size, 5); i++) {
        LOG_PRINT("softmaxOut[%ld] is: %f\n", i, resultData[i]);
    }
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

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t batchSize = 1;
    const int64_t t1 = 16;
    const int64_t t2 = 4;
    const int64_t numHeadsQ = 8;
    const int64_t numHeadsK = 1;
    const int64_t headDim = 128;
    const int64_t maskMode = 3;
    const int64_t cmpRatio = 1;

    std::vector<int64_t> qShape = {t1, numHeadsQ, headDim};
    std::vector<int64_t> kShape = {t2, numHeadsK, headDim};
    std::vector<int64_t> wShape = {t1, numHeadsQ};
    std::vector<int64_t> l1NormShape = {t1, numHeadsK, t2};
    std::vector<int64_t> softmaxLseShape = {numHeadsK, t1};
    std::vector<int64_t> softmaxShape = {t1, numHeadsK, t2};
    std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
    std::vector<int64_t> metadataShape = {64};

    std::vector<uint16_t> qHostData(GetShapeSize(qShape), 0x3C00);
    std::vector<uint16_t> kHostData(GetShapeSize(kShape), 0x3C00);
    std::vector<float> wHostData(GetShapeSize(wShape), 1.0f);
    std::vector<float> attnSoftmaxL1NormHostData(GetShapeSize(l1NormShape), 0.0f);
    for (int64_t t = 0; t < t1; ++t) {
        auto validK = std::min<int64_t>(std::max<int64_t>(1, (t + 1) / cmpRatio), t2);
        for (int64_t kIdx = 0; kIdx < validK; ++kIdx) {
            attnSoftmaxL1NormHostData[t * numHeadsK * t2 + kIdx] = 1.0f / static_cast<float>(validK);
        }
    }
    std::vector<float> softmaxLseHostData(GetShapeSize(softmaxLseShape), 0.0f);
    std::vector<int32_t> cuSeqLensQHostData = {0, static_cast<int32_t>(t1)};
    std::vector<int32_t> cuSeqLensKHostData = {0, static_cast<int32_t>(t2)};
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);
    std::vector<uint16_t> dqHostData(GetShapeSize(qShape), 0);
    std::vector<uint16_t> dkHostData(GetShapeSize(kShape), 0);
    std::vector<float> dwHostData(GetShapeSize(wShape), 0.0f);
    std::vector<float> softmaxOutHostData(GetShapeSize(softmaxShape), 0.0f);

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *wDeviceAddr = nullptr;
    void *attnSoftmaxL1NormDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;
    void *cuSeqLensQDeviceAddr = nullptr;
    void *cuSeqLensKDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *dqDeviceAddr = nullptr;
    void *dkDeviceAddr = nullptr;
    void *dwDeviceAddr = nullptr;
    void *softmaxOutDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *w = nullptr;
    aclTensor *attnSoftmaxL1Norm = nullptr;
    aclTensor *softmaxLse = nullptr;
    aclTensor *cuSeqLensQ = nullptr;
    aclTensor *cuSeqLensK = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dw = nullptr;
    aclTensor *softmaxOut = nullptr;

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT, &w);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnSoftmaxL1NormHostData, l1NormShape, &attnSoftmaxL1NormDeviceAddr, aclDataType::ACL_FLOAT,
                          &attnSoftmaxL1Norm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(cuSeqLensQHostData, cuSeqLensShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(cuSeqLensKHostData, cuSeqLensShape, &cuSeqLensKDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dwHostData, wShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxOutHostData, softmaxShape, &softmaxOutDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    char layoutQ[4] = {'T', 'N', 'D', 0};
    char layoutK[4] = {'T', 'N', 'D', 0};

    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    ret = aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, t1, t2, numHeadsQ, numHeadsK, headDim, layoutQ,
        layoutK, maskMode, cmpRatio, metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDenseLightningIndexerKLLossGradMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                       stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret);
              return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize(
        q, k, w, attnSoftmaxL1Norm, softmaxLse, cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, metadata, layoutQ,
        layoutK, maskMode, cmpRatio, dq, dk, dw, softmaxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDenseLightningIndexerKLLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerKLLossGrad failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    PrintOutResult(softmaxShape, &softmaxOutDeviceAddr);

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(w);
    aclDestroyTensor(attnSoftmaxL1Norm);
    aclDestroyTensor(softmaxLse);
    aclDestroyTensor(cuSeqLensQ);
    aclDestroyTensor(cuSeqLensK);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dw);
    aclDestroyTensor(softmaxOut);

    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(wDeviceAddr);
    aclrtFree(attnSoftmaxL1NormDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(cuSeqLensQDeviceAddr);
    aclrtFree(cuSeqLensKDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dkDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(softmaxOutDeviceAddr);
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
