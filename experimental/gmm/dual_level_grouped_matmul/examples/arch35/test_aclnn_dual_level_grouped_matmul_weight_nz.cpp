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
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dual_level_grouped_matmul_weight_nz.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int64_t M = 32;
constexpr int64_t K = 640;
constexpr int64_t N = 128;
constexpr int64_t EXPERTS = 2;
constexpr int64_t GROUP_ROWS = M / EXPERTS;
constexpr int64_t LEVEL0_GROUP_SIZE = 512;
constexpr int64_t LEVEL1_GROUP_SIZE = 32;
constexpr int64_t LEVEL0_GROUPS = (K + LEVEL0_GROUP_SIZE - 1) / LEVEL0_GROUP_SIZE;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr int64_t LEVEL1_PAIRS = K / (MX_SCALE_PAIR * LEVEL1_GROUP_SIZE);
constexpr float SECOND_LEVEL0_SCALE = 2.0F;
constexpr float EXPERT_BIAS_STEP = 16.0F;
constexpr float WEIGHT_LEVEL1_SCALE = 2.0F;
constexpr uint32_t BF16_TO_FP32_SHIFT = 16;
constexpr int64_t NZ_K = 16;
constexpr int64_t NZ_N = 64;
constexpr uint8_t FP4_ONES = 0x22;
constexpr uint8_t E8M0_ONE = 127;
constexpr uint8_t E8M0_TWO = 128;

void Check(int status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, status=" + std::to_string(status));
    }
}

struct Runtime {
    aclrtStream stream = nullptr;
    bool initialized = false;
    bool deviceSet = false;

    ~Runtime()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(DEVICE_ID);
        }
        if (initialized) {
            aclFinalize();
        }
    }

    void Init()
    {
        Check(aclInit(nullptr), "aclInit");
        initialized = true;
        Check(aclrtSetDevice(DEVICE_ID), "aclrtSetDevice");
        deviceSet = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
};

struct DeviceBuffer {
    void* addr = nullptr;

    ~DeviceBuffer()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }

    void Allocate(size_t bytes)
    {
        if (bytes != 0) {
            Check(aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
    }
};

struct Tensor {
    DeviceBuffer buffer;
    aclTensor* desc = nullptr;
    aclTensorList* list = nullptr;

    ~Tensor()
    {
        if (list != nullptr) {
            aclDestroyTensorList(list);
        } else if (desc != nullptr) {
            aclDestroyTensor(desc);
        }
    }

    template <typename T>
    void Create(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype,
                aclFormat format = ACL_FORMAT_ND, const std::vector<int64_t>& nzShape = {})
    {
        const size_t bytes = host.size() * sizeof(T);
        buffer.Allocate(bytes);
        Check(aclrtMemcpy(buffer.addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        const auto& storage = nzShape.empty() ? shape : nzShape;
        desc = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format, storage.data(),
                               storage.size(), buffer.addr);
        if (desc == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
    }

    aclTensorList* AsList()
    {
        list = aclCreateTensorList(&desc, 1);
        if (list == nullptr) {
            throw std::runtime_error("aclCreateTensorList failed");
        }
        return list;
    }
};

void CheckOutput(const Tensor& y)
{
    std::vector<uint16_t> output(M * N);
    const size_t bytes = output.size() * sizeof(uint16_t);
    Check(aclrtMemcpy(output.data(), bytes, y.buffer.addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
    // Level1 contributes a factor of 2. The two level0 products are 1 and 4.
    constexpr float matmul =
        WEIGHT_LEVEL1_SCALE * (LEVEL0_GROUP_SIZE + (K - LEVEL0_GROUP_SIZE) * SECOND_LEVEL0_SCALE * SECOND_LEVEL0_SCALE);
    for (int64_t row = 0; row < M; ++row) {
        const float expected = matmul + EXPERT_BIAS_STEP * (row / GROUP_ROWS + 1);
        for (int64_t col = 0; col < N; ++col) {
            const uint32_t bits = static_cast<uint32_t>(output[row * N + col]) << BF16_TO_FP32_SHIFT;
            float actual;
            std::memcpy(&actual, &bits, sizeof(actual));
            if (actual != expected) {
                throw std::runtime_error("Output mismatch at [" + std::to_string(row) + ", " + std::to_string(col) +
                                         "], expected " + std::to_string(expected) + ", got " + std::to_string(actual));
            }
        }
    }
    std::cout << "DualLevelGroupedMatmulWeightNz passed, BF16 shape (" << M << ", " << N << ")\n";
}

void RunExample(aclrtStream stream)
{
    Tensor x, weight, xLevel0, xLevel1, weightLevel0, weightLevel1, groupList, bias, y;
    x.Create(std::vector<uint8_t>(M * K / FP4_ELEMENTS_PER_BYTE, FP4_ONES), {M, K}, ACL_FLOAT4_E2M1);
    // Every FP4 element is 1, so the NZ buffer can be filled directly in storage order.
    weight.Create(std::vector<uint8_t>(EXPERTS * K * N / FP4_ELEMENTS_PER_BYTE, FP4_ONES), {EXPERTS, K, N},
                  ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {EXPERTS, N / NZ_N, K / NZ_K, NZ_K, NZ_N});

    std::vector<float> xLevel0Host(M * LEVEL0_GROUPS, 1.0F);
    for (int64_t row = 0; row < M; ++row) {
        xLevel0Host[row * LEVEL0_GROUPS + 1] = SECOND_LEVEL0_SCALE;
    }
    std::vector<float> weightLevel0Host(EXPERTS * LEVEL0_GROUPS * N, 1.0F);
    std::vector<float> biasHost(EXPERTS * N, EXPERT_BIAS_STEP);
    for (int64_t expert = 0; expert < EXPERTS; ++expert) {
        for (int64_t col = 0; col < N; ++col) {
            weightLevel0Host[(expert * LEVEL0_GROUPS + 1) * N + col] = SECOND_LEVEL0_SCALE;
            biasHost[expert * N + col] = EXPERT_BIAS_STEP * (expert + 1);
        }
    }
    xLevel0.Create(xLevel0Host, {M, LEVEL0_GROUPS}, ACL_FLOAT);
    xLevel1.Create(std::vector<uint8_t>(M * LEVEL1_PAIRS * MX_SCALE_PAIR, E8M0_ONE), {M, LEVEL1_PAIRS, MX_SCALE_PAIR},
                   ACL_FLOAT8_E8M0);
    weightLevel0.Create(weightLevel0Host, {EXPERTS, LEVEL0_GROUPS, N}, ACL_FLOAT);
    weightLevel1.Create(std::vector<uint8_t>(EXPERTS * LEVEL1_PAIRS * N * MX_SCALE_PAIR, E8M0_TWO),
                        {EXPERTS, LEVEL1_PAIRS, N, MX_SCALE_PAIR}, ACL_FLOAT8_E8M0);
    groupList.Create(std::vector<int64_t>{GROUP_ROWS, GROUP_ROWS}, {EXPERTS}, ACL_INT64);
    bias.Create(biasHost, {EXPERTS, N}, ACL_FLOAT);
    y.Create(std::vector<uint16_t>(M * N, 0), {M, N}, ACL_BF16);

    constexpr int64_t splitItem = 3;
    constexpr int64_t groupType = 0;
    constexpr int64_t groupListType = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(
              x.AsList(), weight.AsList(), bias.AsList(), weightLevel1.AsList(), xLevel1.AsList(), groupList.desc,
              xLevel0.AsList(), weightLevel0.AsList(), LEVEL0_GROUP_SIZE, LEVEL1_GROUP_SIZE, splitItem, groupType,
              groupListType, nullptr, y.AsList(), &workspaceSize, &executor),
          "aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize");
    std::unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> plan(executor, aclDestroyAclOpExecutor);
    DeviceBuffer workspace;
    workspace.Allocate(workspaceSize);
    const auto runStatus = aclnnDualLevelGroupedMatmulWeightNz(workspace.addr, workspaceSize, plan.release(), stream);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    Check(runStatus, "aclnnDualLevelGroupedMatmulWeightNz");
    Check(syncStatus, "aclrtSynchronizeStream");
    CheckOutput(y);
}
} // namespace

int main()
{
    try {
        Runtime runtime;
        runtime.Init();
        RunExample(runtime.stream);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
