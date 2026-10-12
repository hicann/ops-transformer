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
 * \file test_aclnn_quant_lightning_indexer.cpp
 * \brief
 */
// testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* queryDequantScaleDeviceAddr = nullptr;
    void* keyDequantScaleDeviceAddr = nullptr;
    void* actualSeqLengthsKeyDeviceAddr = nullptr;
    void* blockTableDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* queryDequantScaleTensor = nullptr;
    aclTensor* keyDequantScaleTensor = nullptr;
    aclTensor* actualSeqLengthsKeyTensor = nullptr;
    aclTensor* blockTableTensor = nullptr;
    aclTensor* outTensor = nullptr;
};

int InitializeTensors(TensorResources& resources)
{
    std::vector<int64_t> queryShape = {2, 64, 16, 128};
    std::vector<int64_t> keyShape = {32, 16, 1, 128};
    std::vector<int64_t> weightsShape = {2, 64, 16};
    std::vector<int64_t> queryDequantScaleShape = {2, 64, 16};
    std::vector<int64_t> keyDequantScaleShape = {32, 16, 1};
    std::vector<int64_t> actualSeqLengthsKeyShape = {2};
    std::vector<int64_t> blockTableShape = {2, 32};
    std::vector<int64_t> outShape = {2, 64, 1, 2048};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t queryDequantScaleShapeSize = GetShapeSize(queryDequantScaleShape);
    int64_t keyDequantScaleShapeSize = GetShapeSize(keyDequantScaleShape);
    int64_t actualSeqLengthsKeyShapeSize = GetShapeSize(actualSeqLengthsKeyShape);
    int64_t blockTableShapeSize = GetShapeSize(blockTableShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<int8_t> queryHostData(queryShapeSize, 1);
    std::vector<int8_t> keyHostData(keyShapeSize, 1);
    std::vector<uint16_t> weightsHostData(weightsShapeSize, 0x3C00);
    std::vector<aclFloat16> queryDequantScaleHostData(queryDequantScaleShapeSize, 1.0f);
    std::vector<aclFloat16> keyDequantScaleHostData(keyDequantScaleShapeSize, 1.0f);
    std::vector<int32_t> actualSeqLengthsKeyHostData = {256, 512};
    std::vector<int32_t> blockTableHostData(blockTableShapeSize, 0);
    for (int32_t i = 0; i < 32; i++) {
        blockTableHostData[i] = i;
        blockTableHostData[32 + i] = i;
    }
    std::vector<int32_t> outHostData(outShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_INT8,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_INT8, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(queryDequantScaleHostData, queryDequantScaleShape, &resources.queryDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.queryDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, keyDequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret =
        CreateAclTensor(actualSeqLengthsKeyHostData, actualSeqLengthsKeyShape, &resources.actualSeqLengthsKeyDeviceAddr,
                        aclDataType::ACL_INT32, &resources.actualSeqLengthsKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &resources.blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret =
        CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr, aclDataType::ACL_INT32, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexer(TensorResources& resources, aclrtStream stream, void** workspaceAddr,
                                 uint64_t* workspaceSize)
{
    int64_t queryQuantMode = 0;
    int64_t keyQuantMode = 0;
    int64_t sparseCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "PA_BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, resources.queryDequantScaleTensor,
        resources.keyDequantScaleTensor, nullptr, resources.actualSeqLengthsKeyTensor, resources.blockTableTensor,
        queryQuantMode, keyQuantMode, layoutQuery, layoutKey, sparseCount, sparseMode, preTokens, nextTokens,
        resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.queryDequantScaleTensor) {
        aclDestroyTensor(resources.queryDequantScaleTensor);
    }
    if (resources.keyDequantScaleTensor) {
        aclDestroyTensor(resources.keyDequantScaleTensor);
    }
    if (resources.actualSeqLengthsKeyTensor) {
        aclDestroyTensor(resources.actualSeqLengthsKeyTensor);
    }
    if (resources.blockTableTensor) {
        aclDestroyTensor(resources.blockTableTensor);
    }
    if (resources.outTensor) {
        aclDestroyTensor(resources.outTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.queryDequantScaleDeviceAddr) {
        aclrtFree(resources.queryDequantScaleDeviceAddr);
    }
    if (resources.keyDequantScaleDeviceAddr) {
        aclrtFree(resources.keyDequantScaleDeviceAddr);
    }
    if (resources.actualSeqLengthsKeyDeviceAddr) {
        aclrtFree(resources.actualSeqLengthsKeyDeviceAddr);
    }
    if (resources.blockTableDeviceAddr) {
        aclrtFree(resources.blockTableDeviceAddr);
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
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {2, 64, 1, 2048};
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
    ret = ExecuteQuantLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
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
    PrintOutResult(outShape, &resources.outDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
