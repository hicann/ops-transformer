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
 * \file test_aclnn_fused_causal_conv1d_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_causal_conv1d_v2.h"

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

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// IEEE 754 fp16 decode (uint16_t → float)
float Fp16ToFloat(uint16_t raw)
{
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    float val;
    if (exp == 0) {
        // subnormal or zero
        val = (sign ? -1.0f : 1.0f) * (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        // infinity or NaN
        val = (mant == 0) ? (sign ? -HUGE_VALF : HUGE_VALF) : NAN;
    } else {
        val = (sign ? -1.0f : 1.0f) * (1.0f + static_cast<float>(mant) / 1024.0f) *
              ldexpf(1.0f, static_cast<int>(exp) - 15);
    }
    return val;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
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

    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0, format,
                              shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

void FreeTensorAndBuffer(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t K = 3;                // kernel width
    int64_t dim = 128;            // feature dimension
    int64_t batch = 4;            // number of sequences
    int64_t numSlots = 8;         // total cache slots (>= batch for non-APC)
    int64_t stateLen = 6 + K - 1; // cache state length per slot (decode:m+k-1)
    int64_t cuSeqLen = 19;

    // ---- Tensor shapes ----
    std::vector<int64_t> xShape = {cuSeqLen, dim};
    std::vector<int64_t> weightShape = {K, dim};
    std::vector<int64_t> convStatesShape = {numSlots, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batch + 1};
    std::vector<int64_t> cacheIndicesShape = {batch}; // 1D: non-APC
    std::vector<int64_t> initialStateModeShape = {batch};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> numAcceptedTokensShape = {batch};
    std::vector<int64_t> yShape = {cuSeqLen, dim};

    // ---- Host data ----
    std::vector<uint16_t> hostX(cuSeqLen * dim, 0x3C00); // fp16 1.0

    // weight: pass-through at k=0, zero elsewhere — 用于简化 golden 验证
    std::vector<uint16_t> hostWeight(K * dim, 0);
    for (int64_t d = 0; d < dim; d++) {
        hostWeight[0 * dim + d] = 0x3C00; // k=0: 1.0
    }

    // conv_states: zero-initialized, shape [8, 2, 128]
    std::vector<uint16_t> hostConvStates(numSlots * stateLen * dim, 0);

    // query_start_loc: [0, 5, 8, 15, 19]
    std::vector<int32_t> hostQueryStartLoc = {0, 5, 8, 15, 19};

    // cache_indices: each batch uses a distinct cache slot
    std::vector<int32_t> hostCacheIndices = {0, 3, 1, 5};

    // initial_state_mode: 1=from cache, 0=zero-init, 2=from cache (MTP variant)
    std::vector<int32_t> hostInitialStateMode = {1, 0, 2, 1};

    // bias: zero (unused)
    std::vector<uint16_t> hostBias(dim, 0);

    // num_accepted_tokens: 0 for prefill (no speculative decoding)
    std::vector<int32_t> hostNumAcceptedTokens(batch, 0);

    // y output buffer
    std::vector<uint16_t> hostY(cuSeqLen * dim, 0);

    // ---- Device pointers and aclTensors ----
    void *xDev = nullptr, *weightDev = nullptr, *convStatesDev = nullptr;
    void *queryStartLocDev = nullptr, *cacheIndicesDev = nullptr;
    void *initialStateModeDev = nullptr, *biasDev = nullptr;
    void *numAcceptedTokensDev = nullptr, *yDev = nullptr;

    aclTensor *xTensor = nullptr, *weightTensor = nullptr, *convStatesTensor = nullptr;
    aclTensor *queryStartLocTensor = nullptr, *cacheIndicesTensor = nullptr;
    aclTensor *initialStateModeTensor = nullptr, *biasTensor = nullptr;
    aclTensor *numAcceptedTokensTensor = nullptr, *yTensor = nullptr;

    // ---- Create required tensors ----
    ret = CreateAclTensor(hostX, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostWeight, weightShape, &weightDev, ACL_FLOAT16, &weightTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(hostConvStates, convStatesShape, &convStatesDev, ACL_FLOAT16, &convStatesTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostQueryStartLoc, queryStartLocShape, &queryStartLocDev, ACL_INT32, &queryStartLocTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostCacheIndices, cacheIndicesShape, &cacheIndicesDev, ACL_INT32, &cacheIndicesTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostInitialStateMode, initialStateModeShape, &initialStateModeDev, ACL_INT32,
                          &initialStateModeTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostBias, biasShape, &biasDev, ACL_FLOAT16, &biasTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostNumAcceptedTokens, numAcceptedTokensShape, &numAcceptedTokensDev, ACL_INT32,
                          &numAcceptedTokensTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostY, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t activationMode = 0; // 0: None
    int64_t padSlotId = -1;     // -1: no pad-slot skipping
    int64_t runMode = 0;        // 0: prefill
    int64_t maxQueryLen = 7;
    int64_t residualConnection = 0; // 0: no residual
    int64_t blockSize = 0;          // 0: non-APC
    int64_t convMode = 0;           // 0: Qwen/Pangu7B
    int64_t maxDraftTokens = 7;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnFusedCausalConv1dV2GetWorkspaceSize(
        xTensor, weightTensor, convStatesTensor, queryStartLocTensor, cacheIndicesTensor, initialStateModeTensor,
        biasTensor, numAcceptedTokensTensor,
        nullptr, // num_computed_tokens (non-APC: nullptr)
        nullptr, // block_idx_first_scheduled_token
        nullptr, // block_idx_last_scheduled_token
        nullptr, // initial_state_idx
        activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, maxDraftTokens,
        yTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCausalConv1dV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFusedCausalConv1dV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCausalConv1dV2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto ySize = GetShapeSize(yShape);
    std::vector<uint16_t> yResult(ySize, 0);
    ret = aclrtMemcpy(yResult.data(), ySize * sizeof(uint16_t), yDev, ySize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    auto csSize = GetShapeSize(convStatesShape);
    std::vector<uint16_t> csResult(csSize, 0);
    ret = aclrtMemcpy(csResult.data(), csSize * sizeof(uint16_t), convStatesDev, csSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy convStates from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("=== y output (first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < ySize; i++) {
        LOG_PRINT("  y[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(yResult[i]));
    }
    LOG_PRINT("  ...\n");

    LOG_PRINT("=== convStates output (first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < csSize; i++) {
        LOG_PRINT("  convStates[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(csResult[i]));
    }
    LOG_PRINT("  ...\n");

    FreeTensorAndBuffer(xTensor, xDev);
    FreeTensorAndBuffer(weightTensor, weightDev);
    FreeTensorAndBuffer(convStatesTensor, convStatesDev);
    FreeTensorAndBuffer(queryStartLocTensor, queryStartLocDev);
    FreeTensorAndBuffer(cacheIndicesTensor, cacheIndicesDev);
    FreeTensorAndBuffer(initialStateModeTensor, initialStateModeDev);
    FreeTensorAndBuffer(biasTensor, biasDev);
    FreeTensorAndBuffer(numAcceptedTokensTensor, numAcceptedTokensDev);
    FreeTensorAndBuffer(yTensor, yDev);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
