/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_v2.h"
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

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *cmpResidualKDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *sparseValuesDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *weightsTensor = nullptr;
    aclTensor *cmpResidualKTensor = nullptr;
    aclTensor *sparseIndicesTensor = nullptr;
    aclTensor *sparseValuesTensor = nullptr;
    // candidate (two-level topk)：off 模式输出为 (0,) 空 tensor（预留接口占位）
    aclTensor *candidateTopkIndicesTensor = nullptr;
    aclTensor *candidateBlockLengthTensor = nullptr;
    // candidate source 模式输出：on 时为 [B,S1,N2,candTopkBlocks] 实 tensor（block_length 仍恒空）
    void *candidateSrcIndicesDeviceAddr = nullptr;
    aclTensor *candidateSrcIndicesTensor = nullptr;
};

int CreateEmptyAclTensor(aclTensor **tensor)
{
    std::vector<int64_t> shape = {0};
    std::vector<int64_t> strides = {1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    return 0;
}

int InitializeTensors(TensorResources &resources)
{
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N1 = 16;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t cmpRatio = 4;
    int64_t S2Orig = 130;
    int64_t S2Compressed = S2Orig / cmpRatio;
    int64_t resK = S2Orig % cmpRatio;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2Compressed, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> cmpResidualKShape = {B};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> cmpResidualKHostData(B, static_cast<int32_t>(resK));
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

    ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &resources.cmpResidualKDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cmpResidualKTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 【R4/R11 清理 2026-09-23】metadata（AICPU 前置分核，arch35 协议遗留）示例段移除：
    // arch22 kernel 不消费 metadata，host 对非空 metadata 硬拒绝（R18 定版）
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

    ret = CreateEmptyAclTensor(&resources.candidateTopkIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateEmptyAclTensor(&resources.candidateBlockLengthTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

// candidate source 模式输出构造：candidateTopkIndicesOut 为 [B,S1,N2,candTopkBlocks] int32
// （block_length 恒空，复用 off 模式的 (0,) 占位 tensor）。仅在 source 调用前创建。
int InitializeCandidateSourceTensors(TensorResources &resources)
{
    constexpr int64_t CAND_TOPK_BLOCKS = 64; // (0, 2048] 且 64 的倍数
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N2 = 1;
    std::vector<int64_t> candidateIndicesShape = {B, S1, N2, CAND_TOPK_BLOCKS};
    int64_t candidateIndicesSize = GetShapeSize(candidateIndicesShape);
    std::vector<int32_t> candidateIndicesHostData(candidateIndicesSize, 0);
    return CreateAclTensor(candidateIndicesHostData, candidateIndicesShape, &resources.candidateSrcIndicesDeviceAddr,
                           aclDataType::ACL_INT32, &resources.candidateSrcIndicesTensor);
}

int ExecuteLightningIndexerV2(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                              uint64_t *workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 3;
    int64_t cmpRatio = 4;
    int64_t returnValue = 0;
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

    int ret = aclnnLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        resources.cmpResidualKTensor, nullptr, nullptr, nullptr /* metadata: arch22 不消费且非空即拒 */, topk,
        maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue,
        static_cast<int64_t>(-1) /* candidate_topk_blocks: off */, static_cast<int64_t>(8) /* candidate_block_size */,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, resources.candidateTopkIndicesTensor,
        resources.candidateBlockLengthTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

// candidate (two-level topk) source 模式调用示例：candidate_topk_blocks=64（(0,2048] 且 64 的倍数）
// 时开启，candidate_topk_indices 输出候选块索引（块号或 -1、无序、相对块号），
// candidate_block_length 为预留接口恒空。candidate 仅 ascend910b/ascend910_93（DAV_2201）支持，
// arch35（ascend950）会被 host 校验拒绝，调用方需按 arch 门控。
int ExecuteLightningIndexerV2CandidateSource(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                                             uint64_t *workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 3;
    int64_t cmpRatio = 4;
    int64_t returnValue = 0;          // candidate 与 return_value 互斥，固定 0
    int64_t candidateTopkBlocks = 64; // source on：(0, 2048] 且 64 的倍数
    int64_t candidateBlockSize = 8;   // [2, 64] 且 2 的幂（缺省 8）
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

    int ret = aclnnLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        resources.cmpResidualKTensor, nullptr, nullptr, nullptr /* metadata: arch22 不消费且非空即拒 */, topk,
        maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue, candidateTopkBlocks, candidateBlockSize,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, resources.candidateSrcIndicesTensor,
        resources.candidateBlockLengthTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2GetWorkspaceSize(candidate source) failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2(candidate source) failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const char *name, std::vector<int64_t> &shape, void **deviceAddr)
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
        LOG_PRINT("%s result[%ld] is: %d\n", name, i, resultData[i]);
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
    if (resources.cmpResidualKTensor) {
        aclDestroyTensor(resources.cmpResidualKTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.candidateTopkIndicesTensor) {
        aclDestroyTensor(resources.candidateTopkIndicesTensor);
    }
    if (resources.candidateBlockLengthTensor) {
        aclDestroyTensor(resources.candidateBlockLengthTensor);
    }

    if (resources.candidateSrcIndicesTensor) {
        aclDestroyTensor(resources.candidateSrcIndicesTensor);
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
    if (resources.cmpResidualKDeviceAddr) {
        aclrtFree(resources.cmpResidualKDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
    }
    if (resources.candidateSrcIndicesDeviceAddr) {
        aclrtFree(resources.candidateSrcIndicesDeviceAddr);
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
    // 【R4 清理 2026-09-23】experimental 版定版 ascend910b/ascend910_93（arch22）；
    // ascend950 支持由顶层 attention/lightning_indexer_v2 承担，本示例不再分支
    const NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t candidateTopkBlocks = 64; // source 示例：(0, 2048] 且 64 的倍数
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> candidateIndicesShape = {B, S1, N2, candidateTopkBlocks};
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
    // 1) off 模式回归：candidate_topk_blocks=-1，candidate 输出传空 tensor（旧调用点兼容）
    ret = ExecuteLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult("sparse_indices", sparseIndicesShape, &resources.sparseIndicesDeviceAddr);

    // 释放 off 模式 workspace，source 模式按自身 GetWorkspaceSize 重新分配
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
        workspaceSize = 0;
    }

    // 2) candidate source 模式示例（本算子 candidate 仅此 arch22 支持）
    {
        ret = InitializeCandidateSourceTensors(resources);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("InitializeCandidateSourceTensors failed. ERROR: %d\n", ret);
            CleanupResources(resources, workspaceAddr, stream, deviceId);
            return ret;
        }
        ret = ExecuteLightningIndexerV2CandidateSource(resources, stream, &workspaceAddr, &workspaceSize);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("ExecuteLightningIndexerV2CandidateSource failed. ERROR: %d\n", ret);
            CleanupResources(resources, workspaceAddr, stream, deviceId);
            return ret;
        }

        ret = aclrtSynchronizeStream(stream);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            CleanupResources(resources, workspaceAddr, stream, deviceId);
            return ret;
        }

        PrintOutResult("candidate_topk_indices", candidateIndicesShape, &resources.candidateSrcIndicesDeviceAddr);
    }

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
