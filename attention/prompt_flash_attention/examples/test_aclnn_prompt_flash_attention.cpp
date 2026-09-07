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
 * \file test_prompt_flash_attention_v3.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
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
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *attenDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *attenTensor = nullptr;
    aclTensor *outTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
};

int InitializeTensors(TensorResources &resources)
{
    std::vector<int64_t> queryShape = {1, 2, 1, 16};
    std::vector<int64_t> keyShape = {1, 2, 2, 16};
    std::vector<int64_t> valueShape = {1, 2, 2, 16};
    std::vector<int64_t> attenShape = {1, 1, 1, 2};
    std::vector<int64_t> outShape = {1, 2, 1, 16};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t attenShapeSize = GetShapeSize(attenShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
    std::vector<int8_t> attenHostData(attenShapeSize, 1);
    std::vector<op::fp16_t> outHostData(outShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_FLOAT16,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(attenHostData, attenShape, &resources.attenDeviceAddr, aclDataType::ACL_BOOL,
                          &resources.attenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    std::vector<int64_t> actualSeqlenVector = {2};
    resources.actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    return ACL_SUCCESS;
}

int ExecutePromptFlashAttention(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                                uint64_t *workspaceSize)
{
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1 / sqrt(2);
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;

    constexpr const char LAYER_OUT_STR[] = "BNSD";
    constexpr size_t LAYER_OUT_LEN = sizeof(LAYER_OUT_STR);
    char layerOut[LAYER_OUT_LEN];
    memcpy_s(layerOut, LAYER_OUT_LEN, LAYER_OUT_STR, LAYER_OUT_LEN);

    aclOpExecutor *executor;
    int ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.valueTensor, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads,
        sparseMode, innerPrecise, resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnPromptFlashAttentionV3(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int ProcessResults(TensorResources &resources, const std::vector<int64_t> &outShape)
{
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);

    int ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), resources.outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources &resources, void *workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
        aclDestroyTensor(resources.valueTensor);
    }
    if (resources.attenTensor) {
        aclDestroyTensor(resources.attenTensor);
    }
    if (resources.outTensor) {
        aclDestroyTensor(resources.outTensor);
    }
    if (resources.actualSeqLengths) {
        aclDestroyIntArray(resources.actualSeqLengths);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
        aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.attenDeviceAddr) {
        aclrtFree(resources.attenDeviceAddr);
    }
    if (resources.outDeviceAddr) {
        aclrtFree(resources.outDeviceAddr);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }

    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 2, 1, 16};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecutePromptFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    ret = ProcessResults(resources, outShape);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
