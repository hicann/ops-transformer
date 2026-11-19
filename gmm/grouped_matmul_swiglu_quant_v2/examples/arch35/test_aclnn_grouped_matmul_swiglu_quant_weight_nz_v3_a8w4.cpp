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
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

namespace {
constexpr int64_t EXPERTS = 2;
constexpr int64_t TOKENS = 33;
constexpr int64_t INPUT_K = 256;
constexpr int64_t OUTPUT_N = 256;
constexpr int64_t PAIR = 2;
constexpr int64_t SCALE_BLOCK = 64;
constexpr int64_t NZ_K = 16;
constexpr int64_t NZ_N = 64;
constexpr int64_t SCALE_K = INPUT_K / SCALE_BLOCK;
constexpr int64_t MX_DEQUANT = 2;
constexpr int64_t SPLIT_SWIGLU = 2;
constexpr uint8_t PACKED_FP4_ONE = 0x22;
constexpr uint8_t SCALE_ONE = 127;
constexpr uint32_t FP8_MANTISSA_BITS = 3;
constexpr uint32_t FP8_EXPONENT_MASK = 15;
constexpr uint32_t FP8_MANTISSA_MASK = 7;
constexpr int32_t FP8_EXPONENT_BIAS = 7;
constexpr int32_t E8M0_EXPONENT_BIAS = 127;
constexpr uint8_t FP8_ONE = 0x38;
constexpr uint8_t FP8_SIGN_BIT = 0x80;
constexpr double OUTPUT_TOLERANCE = 0.01;
constexpr double CLAMP_LIMIT = 7.0;
constexpr double GLU_ALPHA = 1.702;
constexpr double GLU_BIAS = 1.0;

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
                  aclDataType dtype, aclFormat format, TensorResource& resource,
                  const std::vector<int64_t>& customStrides = {})
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
    if (!customStrides.empty()) {
        strides = customStrides;
    }
    resource.tensor = aclCreateTensor(view.data(), view.size(), dtype, strides.data(), 0, format, storage.data(),
                                      storage.size(), resource.device);
    return resource.tensor != nullptr;
}

bool CreateInputs(TensorResource& x, TensorResource& xScale, TensorResource& groups, TensorResource& weight,
                  TensorResource& weightScale, TensorResource& y, TensorResource& yScale, TensorResource& smooth,
                  bool useSmooth)
{
    std::vector<uint8_t> xHost(TOKENS * INPUT_K, FP8_ONE);
    std::vector<uint8_t> weightHost(EXPERTS * INPUT_K * OUTPUT_N / PAIR, PACKED_FP4_ONE);
    std::vector<uint8_t> xScaleHost(TOKENS * SCALE_K * PAIR, SCALE_ONE);
    std::vector<uint8_t> weightScaleHost(EXPERTS * SCALE_K * OUTPUT_N * PAIR, SCALE_ONE);
    std::vector<int64_t> groupHost = {1, TOKENS - 1};
    std::vector<uint8_t> yHost(TOKENS * OUTPUT_N / PAIR, 0);
    std::vector<uint8_t> yScaleHost(TOKENS * OUTPUT_N / SCALE_BLOCK, 0);
    bool ok =
        CreateTensor(xHost, {TOKENS, INPUT_K}, {TOKENS, INPUT_K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, x) &&
        CreateTensor(xScaleHost, {TOKENS, SCALE_K, PAIR}, {TOKENS, SCALE_K, PAIR}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                     xScale) &&
        CreateTensor(groupHost, {EXPERTS}, {EXPERTS}, ACL_INT64, ACL_FORMAT_ND, groups) &&
        CreateTensor(weightHost, {EXPERTS, INPUT_K, OUTPUT_N}, {EXPERTS, INPUT_K / NZ_N, OUTPUT_N / NZ_K, NZ_K, NZ_N},
                     ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, weight, {INPUT_K * OUTPUT_N, 1, INPUT_K}) &&
        CreateTensor(weightScaleHost, {EXPERTS, SCALE_K, OUTPUT_N, PAIR}, {EXPERTS, OUTPUT_N, SCALE_K, PAIR},
                     ACL_FLOAT8_E8M0, ACL_FORMAT_ND, weightScale,
                     {SCALE_K * OUTPUT_N * PAIR, PAIR, SCALE_K * PAIR, 1}) &&
        CreateTensor(yHost, {TOKENS, OUTPUT_N / PAIR}, {TOKENS, OUTPUT_N / PAIR}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND,
                     y) &&
        CreateTensor(yScaleHost, {TOKENS, OUTPUT_N / PAIR / SCALE_BLOCK, PAIR},
                     {TOKENS, OUTPUT_N / PAIR / SCALE_BLOCK, PAIR}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, yScale);
    if (!ok) {
        return false;
    }
    weight.list = aclCreateTensorList(&weight.tensor, 1);
    weightScale.list = aclCreateTensorList(&weightScale.tensor, 1);
    if (useSmooth) {
        std::vector<float> data(EXPERTS * OUTPUT_N / PAIR);
        for (int64_t e = 0; e < EXPERTS; ++e) {
            for (int64_t n = 0; n < OUTPUT_N / PAIR; ++n) {
                data[e * OUTPUT_N / PAIR + n] = std::ldexp(1.0F, e);
            }
        }
        ok = CreateTensor(data, {EXPERTS, OUTPUT_N / PAIR}, {EXPERTS, OUTPUT_N / PAIR}, ACL_FLOAT, ACL_FORMAT_ND,
                          smooth);
    }
    return ok && weight.list != nullptr && weightScale.list != nullptr;
}

double DecodeFp8(uint8_t value)
{
    const int exponent = (value >> FP8_MANTISSA_BITS) & FP8_EXPONENT_MASK;
    const double mantissa = static_cast<double>(value & FP8_MANTISSA_MASK) / (1U << FP8_MANTISSA_BITS);
    return exponent == 0 ? std::ldexp(mantissa, 1 - FP8_EXPONENT_BIAS) :
                           std::ldexp(1.0 + mantissa, exponent - FP8_EXPONENT_BIAS);
}

bool CheckOutput(const TensorResource& y, const TensorResource& yScale, bool useSmooth)
{
    std::vector<uint8_t> output(TOKENS * OUTPUT_N / PAIR), scales(TOKENS * OUTPUT_N / SCALE_BLOCK);
    auto ret = aclrtMemcpy(output.data(), output.size(), y.device, output.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    auto scaleRet = aclrtMemcpy(scales.data(), scales.size(), yScale.device, scales.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS || scaleRet != ACL_SUCCESS) {
        return false;
    }
    const double expected = CLAMP_LIMIT / (1.0 + std::exp(-GLU_ALPHA * CLAMP_LIMIT)) * (CLAMP_LIMIT + GLU_BIAS);
    constexpr int64_t scaleElements = SCALE_BLOCK / PAIR;
    for (int64_t row = 0; row < TOKENS; ++row) {
        const double factor = useSmooth && row != 0 ? PAIR : 1;
        for (int64_t col = 0; col < OUTPUT_N / PAIR; ++col) {
            const uint8_t value = output[row * OUTPUT_N / PAIR + col];
            const int scale = scales[row * OUTPUT_N / scaleElements / PAIR + col / scaleElements];
            const double actual = std::ldexp(DecodeFp8(value), scale - E8M0_EXPONENT_BIAS);
            if (!std::isfinite(actual) || value >= FP8_SIGN_BIT ||
                std::fabs(actual - expected * factor) > OUTPUT_TOLERANCE) {
                std::printf("Mismatch row=%ld col=%ld actual=%f expected=%f\n", row, col, actual, expected * factor);
                return false;
            }
        }
    }
    std::printf("V3 A8W4 passed: smooth=%d, shape=[%ld,%ld]\n", useSmooth, TOKENS, OUTPUT_N / PAIR);
    return true;
}

bool RunExample(aclrtStream stream, bool useSmooth, int64_t scaleAlg)
{
    TensorResource x, xScale, groups, weight, weightScale, y, yScale, smooth;
    if (!CreateInputs(x, xScale, groups, weight, weightScale, y, yScale, smooth, useSmooth)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.tensor, weight.list, weightScale.list, nullptr, nullptr, xScale.tensor, smooth.tensor, groups.tensor,
        MX_DEQUANT, 0, MX_DEQUANT, 1, nullptr, SPLIT_SWIGLU, CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS, "rint", scaleAlg, 0.0,
        y.tensor, yScale.tensor, &workspaceSize, &executor);
    Workspace workspace;
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.device, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspace.device, workspaceSize, executor, stream);
    }
    // Drain issued work before releasing tensors, including the API error path.
    auto syncRet = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS || syncRet != ACL_SUCCESS) {
        std::printf("Run failed: ACLNN=%d, synchronize=%d\n", ret, syncRet);
        return false;
    }
    return CheckOutput(y, yScale, useSmooth);
}
} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        return 1;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    bool ok = aclrtCreateStream(&stream) == ACL_SUCCESS;
    if (ok) {
        for (int64_t algorithm = 0; algorithm <= 1 && ok; ++algorithm) {
            ok = RunExample(stream, false, algorithm) && RunExample(stream, true, algorithm);
        }
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
