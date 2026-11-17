/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"

#define CHECK_ACL(call) \
    do { \
        const auto status = (call); \
        if (status != ACL_SUCCESS) { \
            std::fprintf(stderr, "%s failed, error=%d\n", #call, status); \
            return status; \
        } \
    } while (0)

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int64_t E = 2;
constexpr int64_t M = 4;
constexpr int64_t K = 64;
constexpr int64_t N = 128;
constexpr int64_t NZ_K_TILE = 16;
constexpr int64_t NZ_N_TILE = 32;
constexpr int64_t K_TILES = (K + NZ_K_TILE - 1) / NZ_K_TILE;
constexpr int64_t N_TILES = (N + NZ_N_TILE - 1) / NZ_N_TILE;
constexpr int64_t OUTPUT_N = N / 2;
constexpr float INPUT_SCALE = 0.0625F;

using DeviceBuffer = std::unique_ptr<void, decltype(&aclrtFree)>;

struct DeviceTensor {
    DeviceBuffer data{nullptr, aclrtFree};
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> tensor{nullptr, aclDestroyTensor};
};

template <typename T>
int CreateTensor(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                 const std::vector<int64_t>& storageShape, aclDataType dtype, aclFormat format, DeviceTensor& dst)
{
    const auto bytes = hostData.size() * sizeof(T);
    void* data = nullptr;
    CHECK_ACL(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    dst.data.reset(data);
    CHECK_ACL(aclrtMemcpy(data, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> strides(viewShape.size(), 1);
    for (size_t dim = viewShape.size() - 1; dim > 0; --dim) {
        strides[dim - 1] = strides[dim] * viewShape[dim];
    }
    dst.tensor.reset(aclCreateTensor(viewShape.data(), viewShape.size(), dtype, strides.data(), 0, format,
                                     storageShape.data(), storageShape.size(), data));
    if (!dst.tensor) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateNdTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dtype,
                   DeviceTensor& dst)
{
    return CreateTensor(hostData, shape, shape, dtype, ACL_FORMAT_ND, dst);
}

std::vector<int8_t> MakeNzWeight()
{
    // Store W[e, k, n] at NZ[e, n / 32, k / 16, k % 16, n % 32].
    std::vector<int8_t> weight(E * N_TILES * K_TILES * NZ_K_TILE * NZ_N_TILE, 0);
    for (int64_t expert = 0; expert < E; ++expert) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t n = 0; n < N; ++n) {
                const auto tile = (expert * N_TILES + n / NZ_N_TILE) * K_TILES + k / NZ_K_TILE;
                const auto index = (tile * NZ_K_TILE + k % NZ_K_TILE) * NZ_N_TILE + n % NZ_N_TILE;
                weight[index] = static_cast<int8_t>((expert + 1) * (n / NZ_N_TILE + 1));
            }
        }
    }
    return weight;
}

int RunExample(aclrtStream stream)
{
    DeviceTensor x;
    DeviceTensor weight;
    DeviceTensor weightScale;
    DeviceTensor xScale;
    DeviceTensor groupList;
    DeviceTensor output;
    DeviceTensor outputScale;
    const std::vector<int8_t> xHost(M * K, 1);
    const auto weightHost = MakeNzWeight();
    const std::vector<float> weightScaleHost(E * N, INPUT_SCALE);
    const std::vector<float> xScaleHost(M, INPUT_SCALE);
    const std::vector<int64_t> groupListHost{M / E, M};
    std::vector<int8_t> outputHost(M * OUTPUT_N, 0);
    std::vector<float> outputScaleHost(M, 0);

    CHECK_ACL(CreateNdTensor(xHost, {M, K}, ACL_INT8, x));
    // The public WeightNZ API on Ascend 950 takes a 3D logical view and 5D NZ storage.
    CHECK_ACL(CreateTensor(weightHost, {E, K, N}, {E, N_TILES, K_TILES, NZ_K_TILE, NZ_N_TILE}, ACL_INT8,
                           ACL_FORMAT_FRACTAL_NZ, weight));
    CHECK_ACL(CreateNdTensor(weightScaleHost, {E, N}, ACL_FLOAT, weightScale));
    CHECK_ACL(CreateNdTensor(xScaleHost, {M}, ACL_FLOAT, xScale));
    CHECK_ACL(CreateNdTensor(groupListHost, {E}, ACL_INT64, groupList));
    CHECK_ACL(CreateNdTensor(outputHost, {M, OUTPUT_N}, ACL_INT8, output));
    CHECK_ACL(CreateNdTensor(outputScaleHost, {M}, ACL_FLOAT, outputScale));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
        x.tensor.get(), weight.tensor.get(), nullptr, nullptr, weightScale.tensor.get(), xScale.tensor.get(),
        groupList.tensor.get(), output.tensor.get(), outputScale.tensor.get(), nullptr, &workspaceSize, &executor));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    const DeviceBuffer workspace{workspaceAddr, aclrtFree};
    CHECK_ACL(aclnnGroupedMatmulSwigluQuantWeightNZ(workspace.get(), workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    const auto outputBytes = outputHost.size() * sizeof(outputHost[0]);
    const auto scaleBytes = outputScaleHost.size() * sizeof(outputScaleHost[0]);
    CHECK_ACL(aclrtMemcpy(outputHost.data(), outputBytes, output.data.get(), outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(
        aclrtMemcpy(outputScaleHost.data(), scaleBytes, outputScale.data.get(), scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t token = 0; token < outputScaleHost.size(); ++token) {
        const auto first = token * OUTPUT_N;
        std::printf("token=%zu output_first=%d output_last=%d outputScale=%.9g\n", token, outputHost[first],
                    outputHost[first + OUTPUT_N - 1], outputScaleHost[token]);
    }
    return ACL_SUCCESS;
}
} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    auto status = aclrtSetDevice(DEVICE_ID);
    if (status == ACL_SUCCESS) {
        aclrtStream stream = nullptr;
        status = aclrtCreateStream(&stream);
        if (status == ACL_SUCCESS) {
            status = RunExample(stream);
            aclrtDestroyStream(stream);
        } else {
            std::fprintf(stderr, "aclrtCreateStream failed, error=%d\n", status);
        }
        aclrtResetDevice(DEVICE_ID);
    } else {
        std::fprintf(stderr, "aclrtSetDevice failed, error=%d\n", status);
    }
    aclFinalize();
    return status;
}
