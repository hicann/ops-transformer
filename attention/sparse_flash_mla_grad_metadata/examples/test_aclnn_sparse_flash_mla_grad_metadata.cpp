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
 * @file test_aclnn_sparse_flash_mla_grad_metadata.cpp
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf(fmt "\n", ##__VA_ARGS__); \
            return (ret_val); \
        } \
    } while (0)

// Constants
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLAG_METADATA_TOTAL_SIZE = 1024;
using SMLAG_METADATA_T = int32_t;
constexpr uint32_t GRAD_METADATA_SIZE = 7;

// Grad Metadata Index Definitions
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;
constexpr uint32_t MAX_ORI_KV_SIZE = 5;
constexpr uint32_t MAX_CMP_KV_SIZE = 6;

struct SmlagMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> onExitScope)
        : m_exitFunc(std::move(onExitScope)),
          m_isDismissed(false)
    {}
    // 禁止拷贝
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void* hostAddr{nullptr};
    void* deviceAddr{nullptr};
    aclTensor* data{nullptr};
};

struct ArgScenario {
    bool hasCuSeq{false};
    bool hasSeqused{false};
};

struct ArgContext {
    // required input
    int64_t numHeadsQ{0};
    int64_t numHeadsKv{0};
    int64_t headDim{0};
    // optional input
    Tensor cuSeqlensQOptional{};
    Tensor cuSeqlensOriKvOptional{};
    Tensor cuSeqlensCmpKvOptional{};
    Tensor sequsedQOptional{};
    Tensor sequsedOriKvOptional{};
    Tensor sequsedCmpKvOptional{};
    Tensor cmpResidualKvOptional{};
    Tensor oriTopkLengthOptional{};
    Tensor cmpTopkLengthOptional{};
    int64_t batchSize{0};
    int64_t maxSeqlenQ{0};
    int64_t maxSeqlenOriKv{0};
    int64_t maxSeqlenCmpKv{0};
    int64_t oriTopk{0};
    int64_t cmpTopk{0};
    int64_t cmpRatio{0};
    int64_t oriMaskMode{0};
    int64_t cmpMaskMode{0};
    int64_t oriWinLeft{-1};
    int64_t oriWinRight{-1};
    char* layoutQOptional{nullptr};
    char* layoutKvOptional{nullptr};
    bool hasOriKv{true};
    bool hasCmpKv{true};
    // output
    Tensor metadata{};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
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

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t>& shape, Tensor& tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.data != nullptr, ACL_ERROR_FAILURE, "aclCreateTensor failed");
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor& tensor)
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

void DestroyArgs(ArgContext& context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario& scenario, ArgContext& context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    ret = CreateTensor(aclDataType::ACL_INT32, {SMLAG_METADATA_TOTAL_SIZE}, context.metadata); // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char*)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char*)malloc(sizeof(char) * 16);
    CHECK_LOG_RET(context.layoutQOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutQOptional failed");
    CHECK_LOG_RET(context.layoutKvOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutKvOptional failed");
    strcpy(context.layoutQOptional, "BSND");  // BSND,TND
    strcpy(context.layoutKvOptional, "BSND"); // BSND,TND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario{};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context{};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnSparseFlashMlaGradMetadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight, context.layoutQOptional,
        context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // 调用aclnnSparseFlashMlaGradMetadata第二段接口
    ret = aclnnSparseFlashMlaGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaGradMetadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    SmlagMetadata result{};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < GRAD_METADATA_SIZE; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Total Num                 : %u\n", result.gradMetadata[TOTAL_NUM]);
        printf("    Former Core Process Num   : %u\n", result.gradMetadata[FORMER_CORE_PROCESS_NUM]);
        printf("    Remain Core Process Num   : %u\n", result.gradMetadata[REMAIN_CORE_PROCESS_NUM]);
        printf("    Remain Core Num           : %u\n", result.gradMetadata[REMAIN_CORE_NUM]);
        printf("    Used Core Num             : %u\n", result.gradMetadata[USED_CORE_NUM]);
        printf("    Max Ori Kv Size           : %u\n", result.gradMetadata[MAX_ORI_KV_SIZE]);
        printf("    Max Cmp Kv Size           : %u\n", result.gradMetadata[MAX_CMP_KV_SIZE]);
    }

    return 0;
}
