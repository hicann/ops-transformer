/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_activation_weight_nz.h"

namespace {
constexpr int64_t EXPERTS = 1;
constexpr int64_t TOKENS = 32;
constexpr int64_t INPUT_K = 128;
constexpr int64_t OUTPUT_N = 128;
constexpr int64_t PAIR = 2;
constexpr int64_t SCALE_BLOCK = 64;
constexpr int64_t NZ_K = 16;
constexpr int64_t NZ_N = 64;
constexpr int64_t SCALE_K = INPUT_K / SCALE_BLOCK;
constexpr int64_t MX_DEQUANT = 2;
constexpr int64_t SPLIT_SWIGLU = 2;
constexpr uint8_t PACKED_FP4_ONE = 0x22;
constexpr uint8_t SCALE_ONE = 127;
constexpr uint32_t BF16_SHIFT = 16;
constexpr double CLAMP_LIMIT = 7.0;
constexpr double GLU_ALPHA = 1.702;
constexpr double GLU_BIAS = 1.0;
constexpr double BF16_TOLERANCE = 0.25;

bool CheckStatus(const char* operation, int32_t status)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", operation, status);
        return false;
    }
    return true;
}

struct TensorResource {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
    aclTensorList* list = nullptr;
    ~TensorResource()
    {
        if (list != nullptr) {
            aclDestroyTensorList(list);
        } else if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (device != nullptr) {
            aclrtFree(device);
        }
    }
};

struct Workspace {
    void* device = nullptr;
    ~Workspace()
    {
        if (device != nullptr) {
            aclrtFree(device);
        }
    }
};

template <typename T>
bool CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& view, const std::vector<int64_t>& storage,
                  aclDataType dtype, aclFormat format, TensorResource& resource)
{
    const size_t bytes = host.size() * sizeof(T);
    auto ret = aclrtMalloc(&resource.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(resource.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("CreateTensor memory operation failed: %d\n", ret);
        return false;
    }
    std::vector<int64_t> strides(view.size(), 1);
    for (size_t i = view.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * view[i];
    }
    resource.tensor = aclCreateTensor(view.data(), view.size(), dtype, strides.data(), 0, format, storage.data(),
                                      storage.size(), resource.device);
    if (resource.tensor == nullptr) {
        std::printf("aclCreateTensor failed: dtype=%d, format=%d, rank=%zu\n", dtype, format, view.size());
    }
    return resource.tensor != nullptr;
}

bool CreateInputs(TensorResource& x, TensorResource& xScale, TensorResource& groups, TensorResource& weight,
                  TensorResource& weightScale, TensorResource& y)
{
    // Every FP4 nibble encodes 1. Constant weights have identical ND and NZ byte contents.
    std::vector<uint8_t> xHost(TOKENS * INPUT_K / PAIR, PACKED_FP4_ONE);
    std::vector<uint8_t> weightHost(EXPERTS * INPUT_K * OUTPUT_N / PAIR, PACKED_FP4_ONE);
    std::vector<uint8_t> xScaleHost(TOKENS * SCALE_K * PAIR, SCALE_ONE);
    std::vector<uint8_t> weightScaleHost(EXPERTS * SCALE_K * OUTPUT_N * PAIR, SCALE_ONE);
    std::vector<int64_t> groupHost = {TOKENS};
    std::vector<uint16_t> yHost(TOKENS * OUTPUT_N / PAIR, 0);
    bool ok =
        CreateTensor(xHost, {TOKENS, INPUT_K}, {TOKENS, INPUT_K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, x) &&
        CreateTensor(xScaleHost, {TOKENS, SCALE_K, PAIR}, {TOKENS, SCALE_K, PAIR}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                     xScale) &&
        CreateTensor(groupHost, {EXPERTS}, {EXPERTS}, ACL_INT64, ACL_FORMAT_ND, groups) &&
        CreateTensor(weightHost, {EXPERTS, INPUT_K, OUTPUT_N}, {EXPERTS, OUTPUT_N / NZ_N, INPUT_K / NZ_K, NZ_K, NZ_N},
                     ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, weight) &&
        CreateTensor(weightScaleHost, {EXPERTS, SCALE_K, OUTPUT_N, PAIR}, {EXPERTS, SCALE_K, OUTPUT_N, PAIR},
                     ACL_FLOAT8_E8M0, ACL_FORMAT_ND, weightScale) &&
        CreateTensor(yHost, {TOKENS, OUTPUT_N / PAIR}, {TOKENS, OUTPUT_N / PAIR}, ACL_BF16, ACL_FORMAT_ND, y);
    if (!ok) {
        return false;
    }
    weight.list = aclCreateTensorList(&weight.tensor, 1);
    weightScale.list = aclCreateTensorList(&weightScale.tensor, 1);
    if (weight.list == nullptr || weightScale.list == nullptr) {
        std::printf("aclCreateTensorList failed: weight=%p, weightScale=%p\n", static_cast<void*>(weight.list),
                    static_cast<void*>(weightScale.list));
    }
    return weight.list != nullptr && weightScale.list != nullptr;
}

bool CheckOutput(const TensorResource& y)
{
    std::vector<uint16_t> output(TOKENS * OUTPUT_N / PAIR);
    const size_t bytes = output.size() * sizeof(uint16_t);
    auto ret = aclrtMemcpy(output.data(), bytes, y.device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::printf("Copy output failed: %d\n", ret);
        return false;
    }
    // Both matmul branches equal K; mode 2 clamps them before applying SwiGLU.
    const double clipped = std::fmin(static_cast<double>(INPUT_K), CLAMP_LIMIT);
    const double expected = clipped / (1.0 + std::exp(-GLU_ALPHA * clipped)) * (clipped + GLU_BIAS);
    for (auto value : output) {
        uint32_t bits = static_cast<uint32_t>(value) << BF16_SHIFT;
        float actual = 0;
        std::memcpy(&actual, &bits, sizeof(actual));
        if (!std::isfinite(actual) || std::fabs(actual - expected) > BF16_TOLERANCE) {
            std::printf("Output mismatch: actual=%f, expected=%f\n", actual, expected);
            return false;
        }
    }
    std::printf("GroupedMatmulActivationWeightNz passed: BF16 shape [%ld, %ld], expected=%f\n", TOKENS, OUTPUT_N / PAIR,
                expected);
    return true;
}

bool RunExample(aclrtStream stream)
{
    TensorResource x, xScale, groups, weight, weightScale, y;
    if (!CreateInputs(x, xScale, groups, weight, weightScale, y)) {
        std::printf("CreateInputs failed\n");
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
        x.tensor, xScale.tensor, groups.tensor, weight.list, weightScale.list, nullptr, MX_DEQUANT, 0, "swiglu", 1,
        nullptr, SPLIT_SWIGLU, CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS, y.tensor, &workspaceSize, &executor);
    Workspace workspace;
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.device, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (!CheckStatus("GetWorkspaceSize or workspace allocation", ret)) {
        if (executor != nullptr) {
            CheckStatus("aclDestroyAclOpExecutor", aclDestroyAclOpExecutor(executor));
        }
        return false;
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnGroupedMatmulActivationWeightNz(workspace.device, workspaceSize, executor, stream);
    }
    // Drain issued work before releasing tensors, including the API error path.
    auto syncRet = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS || syncRet != ACL_SUCCESS) {
        std::printf("Run failed: ACLNN=%d, synchronize=%d\n", ret, syncRet);
        return false;
    }
    return CheckOutput(y);
}
} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    if (!CheckStatus("aclInit", aclInit(nullptr))) {
        return 1;
    }
    if (!CheckStatus("aclrtSetDevice", aclrtSetDevice(deviceId))) {
        CheckStatus("aclFinalize", aclFinalize());
        return 1;
    }
    aclrtStream stream = nullptr;
    bool ok = CheckStatus("aclrtCreateStream", aclrtCreateStream(&stream));
    if (ok) {
        ok = RunExample(stream);
        ok = CheckStatus("aclrtDestroyStream", aclrtDestroyStream(stream)) && ok;
    }
    ok = CheckStatus("aclrtResetDevice", aclrtResetDevice(deviceId)) && ok;
    ok = CheckStatus("aclFinalize", aclFinalize()) && ok;
    return ok ? 0 : 1;
}
