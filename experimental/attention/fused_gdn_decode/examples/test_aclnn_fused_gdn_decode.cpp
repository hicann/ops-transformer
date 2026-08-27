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
#include "aclnnop/aclnn_fused_gdn_decode.h"

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
    constexpr int64_t qkHeads = 1;
    constexpr int64_t valueHeads = 2;
    constexpr int64_t keyDim = 64;
    constexpr int64_t valueDim = 8;
    constexpr int64_t stateSlots = 2;
    constexpr float scale = 0.125f;
    constexpr float softplusThreshold = 20.0f;

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cerr << "ACL init failed: " << ret << std::endl; return ret);

    const std::vector<int64_t> mixedShape = {batch, 2 * qkHeads * keyDim + valueHeads * valueDim};
    const std::vector<int64_t> gateShape = {batch, valueHeads};
    const std::vector<int64_t> paramShape = {valueHeads};
    const std::vector<int64_t> stateShape = {stateSlots, valueHeads, valueDim, keyDim};
    const std::vector<int64_t> indexShape = {batch};
    const std::vector<int64_t> outShape = {batch, 1, valueHeads, valueDim};

    std::vector<aclFloat16> mixedData(Numel(mixedShape), aclFloatToFloat16(0.1f));
    std::vector<aclFloat16> aData(Numel(gateShape), aclFloatToFloat16(0.1f));
    std::vector<aclFloat16> bData(Numel(gateShape), aclFloatToFloat16(0.1f));
    std::vector<float> aLogData(Numel(paramShape), 0.0f);
    std::vector<aclFloat16> dtBiasData(Numel(paramShape), aclFloatToFloat16(0.1f));
    std::vector<float> stateData(Numel(stateShape), 0.1f);
    std::vector<int32_t> stateIndicesData = {1};
    std::vector<aclFloat16> outData(Numel(outShape), aclFloatToFloat16(0.0f));

    std::vector<void *> deviceAddrs(8, nullptr);
    std::vector<aclTensor *> tensors(8, nullptr);
    ret = CreateTensor(mixedData, mixedShape, ACL_FLOAT16, &deviceAddrs[0], &tensors[0]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(aData, gateShape, ACL_FLOAT16, &deviceAddrs[1], &tensors[1]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(bData, gateShape, ACL_FLOAT16, &deviceAddrs[2], &tensors[2]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(aLogData, paramShape, ACL_FLOAT, &deviceAddrs[3], &tensors[3]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(dtBiasData, paramShape, ACL_FLOAT16, &deviceAddrs[4], &tensors[4]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(stateData, stateShape, ACL_FLOAT, &deviceAddrs[5], &tensors[5]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(stateIndicesData, indexShape, ACL_INT32, &deviceAddrs[6], &tensors[6]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateTensor(outData, outShape, ACL_FLOAT16, &deviceAddrs[7], &tensors[7]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFusedGdnDecodeGetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
                                              tensors[6], scale, softplusThreshold, tensors[7], &workspaceSize,
                                              &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, std::cerr << "GetWorkspaceSize failed: " << ret << std::endl; return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnFusedGdnDecode(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, std::cerr << "Execute failed: " << ret << std::endl; return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), deviceAddrs[7],
                      outData.size() * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    for (size_t i = 0; i < std::min<size_t>(outData.size(), 8); ++i) {
        std::cout << "out[" << i << "]=" << aclFloat16ToFloat(outData[i]) << std::endl;
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
