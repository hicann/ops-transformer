/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule_compute_wy.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

namespace {

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dtype,
                 void **deviceAddr, aclTensor **tensor)
{
    const size_t bytes = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                              *deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return aclrtCreateStream(stream);
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t batch = 1;
    constexpr int64_t tokens = 128;
    constexpr int64_t keyHeads = 2;
    constexpr int64_t valueHeads = 4;
    constexpr int64_t keyDim = 64;
    constexpr int64_t valueDim = 64;
    constexpr int64_t chunkSize = 64;

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cerr << "ACL init failed: " << ret << std::endl; return ret);

    const std::vector<int64_t> qShape = {batch, tokens, keyHeads, keyDim};
    const std::vector<int64_t> vShape = {batch, tokens, valueHeads, valueDim};
    const std::vector<int64_t> gShape = {batch, tokens, valueHeads};
    const std::vector<int64_t> qKernelShape = {batch, keyHeads, tokens, keyDim};
    const std::vector<int64_t> wKernelShape = {batch, valueHeads, tokens, keyDim};
    const std::vector<int64_t> uKernelShape = {batch, valueHeads, tokens, valueDim};
    const std::vector<int64_t> gKernelShape = {batch, valueHeads, tokens};

    std::vector<aclFloat16> qData(Numel(qShape), aclFloatToFloat16(0.01f));
    std::vector<aclFloat16> kData(Numel(qShape), aclFloatToFloat16(0.01f));
    std::vector<aclFloat16> vData(Numel(vShape), aclFloatToFloat16(0.01f));
    // The gate is a log-decay, so it must be non-positive.
    std::vector<float> gData(Numel(gShape), -0.01f);
    std::vector<aclFloat16> betaData(Numel(gShape), aclFloatToFloat16(0.2f));
    std::vector<aclFloat16> qKernelData(Numel(qKernelShape), aclFloatToFloat16(0.0f));
    std::vector<aclFloat16> kKernelData(Numel(qKernelShape), aclFloatToFloat16(0.0f));
    std::vector<aclFloat16> wKernelData(Numel(wKernelShape), aclFloatToFloat16(0.0f));
    std::vector<aclFloat16> uKernelData(Numel(uKernelShape), aclFloatToFloat16(0.0f));
    std::vector<float> gKernelData(Numel(gKernelShape), 0.0f);

    std::vector<void *> deviceAddrs(10, nullptr);
    std::vector<aclTensor *> tensors(10, nullptr);
    ret = CreateTensor(qData, qShape, ACL_FLOAT16, &deviceAddrs[0], &tensors[0]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(kData, qShape, ACL_FLOAT16, &deviceAddrs[1], &tensors[1]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(vData, vShape, ACL_FLOAT16, &deviceAddrs[2], &tensors[2]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(gData, gShape, ACL_FLOAT, &deviceAddrs[3], &tensors[3]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(betaData, gShape, ACL_FLOAT16, &deviceAddrs[4], &tensors[4]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(qKernelData, qKernelShape, ACL_FLOAT16, &deviceAddrs[5], &tensors[5]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(kKernelData, qKernelShape, ACL_FLOAT16, &deviceAddrs[6], &tensors[6]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(wKernelData, wKernelShape, ACL_FLOAT16, &deviceAddrs[7], &tensors[7]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(uKernelData, uKernelShape, ACL_FLOAT16, &deviceAddrs[8], &tensors[8]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(gKernelData, gKernelShape, ACL_FLOAT, &deviceAddrs[9], &tensors[9]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                                                            chunkSize, tensors[5], tensors[6], tensors[7], tensors[8],
                                                            tensors[9], &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, std::cerr << "GetWorkspaceSize failed: " << ret << std::endl; return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnChunkGatedDeltaRuleComputeWy(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, std::cerr << "Execute failed: " << ret << std::endl; return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(uKernelData.data(), uKernelData.size() * sizeof(uKernelData[0]), deviceAddrs[8],
                      uKernelData.size() * sizeof(uKernelData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    for (size_t i = 0; i < std::min<size_t>(uKernelData.size(), 8); ++i) {
        std::cout << "u_kernel[" << i << "]=" << aclFloat16ToFloat(uKernelData[i]) << std::endl;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    for (aclTensor *tensor : tensors) {
        aclDestroyTensor(tensor);
    }
    for (void *addr : deviceAddrs) {
        aclrtFree(addr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
