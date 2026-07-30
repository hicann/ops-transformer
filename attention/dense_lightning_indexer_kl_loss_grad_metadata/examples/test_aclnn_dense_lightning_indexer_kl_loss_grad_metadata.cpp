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
 * @file test_aclnn_dense_lightning_indexer_kl_loss_grad_metadata.cpp
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf(fmt "\n", ##__VA_ARGS__); \
            return (ret_val); \
        } \
    } while (0)

constexpr uint32_t SLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t SLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t SLI_METADATA_SIZE = 64;

constexpr uint32_t GRAD_METADATA_SIZE = 5;
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;

struct SliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[SLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[SLI_METADATA_MAX_CORE_NUM];
};

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> onExitScope)
        : m_exitFunc(std::move(onExitScope)),
          m_isDismissed(false)
    {}
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss() { m_isDismissed = true; }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr{nullptr};
    void *deviceAddr{nullptr};
    aclTensor *data{nullptr};
};

struct ArgScenario {
    bool hasCuSeq{true};
};

struct ArgContext {
    Tensor cuSeqLensQOptional{};
    Tensor cuSeqLensKOptional{};
    Tensor seqUsedQOptional{};
    Tensor seqUsedKOptional{};
    Tensor cmpResidualKOptional{};
    Tensor metadata{};
    int64_t batchSize{0};
    int64_t maxSeqLenQ{0};
    int64_t maxSeqLenK{0};
    int64_t numHeadsQ{8};
    int64_t numHeadsK{1};
    int64_t headDim{128};
    char *layoutQOptional{nullptr};
    char *layoutKOptional{nullptr};
    int64_t maskMode{0};
    int64_t cmpRatio{4};
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqLenQ = 16;
    context.maxSeqLenK = 4;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutKOptional, scenario.hasCuSeq ? "TND" : "BSND");

    ret = CreateTensor(aclDataType::ACL_INT32, {SLI_METADATA_SIZE}, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize + 1}, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize + 1}, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, {0, static_cast<int32_t>(context.maxSeqLenQ)});
        SetInt32TensorData(context.cuSeqLensKOptional, {0, static_cast<int32_t>(context.maxSeqLenK)});
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const SliGradKLLossMetaData &metadata)
{
    const int32_t *data = reinterpret_cast<const int32_t *>(&metadata);
    printf("TOTAL_NUM               : %d\n", data[TOTAL_NUM]);
    printf("FORMER_CORE_PROCESS_NUM : %d\n", data[FORMER_CORE_PROCESS_NUM]);
    printf("REMAIN_CORE_PROCESS_NUM : %d\n", data[REMAIN_CORE_PROCESS_NUM]);
    printf("REMAIN_CORE_NUM         : %d\n", data[REMAIN_CORE_NUM]);
    printf("USED_CORE_NUM           : %d\n", data[USED_CORE_NUM]);
    return;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario{};
    scenario.hasCuSeq = true;
    ArgContext context{};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.layoutQOptional,
        context.layoutKOptional, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
                  "aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnDenseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnDenseLightningIndexerKLLossGradMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    SliGradKLLossMetaData result{};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
