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
 * \file test_aclnn_sparse_lightning_indexer.cpp
 * \brief aclnnSparseLightningIndexer 调用示例（consumer，设计 §5.1）：
 * 基本 consumer 调用 + 空 candidate_block_length 传法。仅 ascend910b/ascend910_93。
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer.h"
#include "aclnn/opdev/platform.h"

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

// 空 tensor（candidate_block_length 预留接口的空传法，C7）
int CreateEmptyAclTensor(aclTensor **tensor)
{
    std::vector<int64_t> shape = {0};
    std::vector<int64_t> strides = {1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    return 0;
}

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *candidateTopkIndicesDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *sparseValuesDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *weightsTensor = nullptr;
    aclTensor *candidateTopkIndicesTensor = nullptr;
    aclTensor *candidateBlockLengthTensor = nullptr; // 空 tensor 占位
    aclTensor *sparseIndicesTensor = nullptr;
    aclTensor *sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources &resources)
{
    // BSND/BSND, FP16: B=2, S1=8, N1=8, D=128, S2=64, topk=32, mask_mode=0, cmp_ratio=1
    // candidate: [B, S1, N2=1, candBlocks=64]（S2=64 / blockSize=8 → 8 个块，0..7 为全候选，其余 -1）
    int64_t B = 2;
    int64_t S1 = 8;
    int64_t N1 = 8;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t S2 = 64;
    int64_t topk = 32;
    int64_t candBlocks = 64;
    int64_t blockSize = 8;
    int64_t numBlocks = S2 / blockSize;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> candidateTopkIndicesShape = {B, S1, N2, candBlocks};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t candidateShapeSize = GetShapeSize(candidateTopkIndicesShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> candidateHostData(candidateShapeSize, -1);
    for (int64_t i = 0; i < candidateShapeSize; i++) {
        int64_t slot = i % candBlocks;
        if (slot < numBlocks) {
            candidateHostData[i] = static_cast<int32_t>(slot); // 全候选（覆盖全部块号）
        }
    }
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 0.0f);

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
    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(candidateHostData, candidateTopkIndicesShape, &resources.candidateTopkIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.candidateTopkIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // candidate_block_length：预留接口空传法（shape (0,)）
    ret = CreateEmptyAclTensor(&resources.candidateBlockLengthTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteSparseLightningIndexer(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                                  uint64_t *workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;        // C6：恒 0
    int64_t candidateBlockSize = 8; // [2,64] 2 的幂，默认 8
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
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
    aclOpExecutor *executor;

    // metadata 可选（arch22 kernel 忽略，接口一致性），此处传 nullptr
    int ret = aclnnSparseLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, resources.candidateTopkIndicesTensor, resources.candidateBlockLengthTensor,
        topk, maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue, candidateBlockSize,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
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

void CleanupResources(TensorResources &resources, void *workspaceAddr, aclrtStream stream, int32_t deviceId)
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
    if (resources.candidateTopkIndicesTensor) {
        aclDestroyTensor(resources.candidateTopkIndicesTensor);
    }
    if (resources.candidateBlockLengthTensor) {
        aclDestroyTensor(resources.candidateBlockLengthTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
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
    if (resources.candidateTopkIndicesDeviceAddr) {
        aclrtFree(resources.candidateTopkIndicesDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
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
    // 本算子仅注册 ascend910b/ascend910_93（N6，框架级不支持其他 arch）
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 8;
    int64_t N2 = 1;
    int64_t topk = 32;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    ret = ExecuteSparseLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteSparseLightningIndexer failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
