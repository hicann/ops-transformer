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
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_text_gated_residual.h"

// FP16 zero inputs: mixed and inject must both be zero. No backward cache requested.
namespace {
constexpr int64_t EXAMPLE_SEQUENCE = 32;
constexpr int64_t EXAMPLE_CHANNELS = 4;
constexpr int64_t EXAMPLE_HIDDEN = 512;
constexpr size_t MIXED_ALLOCATION_INDEX = 4;
} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
        return 1;
    std::vector<aclTensor*> tensors;
    std::vector<void*> allocations;
    auto tensor = [&](std::vector<int64_t> shape) -> aclTensor* {
        std::vector<int64_t> strides(shape.size());
        int64_t elements = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = elements;
            elements *= shape[i];
        }
        void* data = nullptr;
        if (aclrtMalloc(&data, elements * 2, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            return nullptr;
        allocations.push_back(data);
        std::vector<uint16_t> zeros(elements, 0);
        if (aclrtMemcpy(data, elements * 2, zeros.data(), elements * 2, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            return nullptr;
        auto* t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0, ACL_FORMAT_ND,
                                  shape.data(), shape.size(), data);
        if (t)
            tensors.push_back(t);
        return t;
    };
    auto* x = tensor({1, 32, 2048});
    auto* down = tensor({64, 2048});
    auto* up = tensor({2048, 64});
    auto* gate = tensor({4, 2048});
    auto* mixed = tensor({1, 32, 512});
    auto* inject = tensor({1, 32, 4});
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    int result = 1;
    void* workspace = nullptr;
    if (x && down && up && gate && mixed && inject &&
        aclnnTextGatedResidualGetWorkspaceSize(x, down, up, gate, nullptr, EXAMPLE_CHANNELS, 1e-6, false, mixed, inject,
                                               nullptr, nullptr, nullptr, &bytes, &executor) == ACL_SUCCESS &&
        (bytes == 0 || aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
        aclnnTextGatedResidual(workspace, bytes, executor, stream) == ACL_SUCCESS &&
        aclrtSynchronizeStream(stream) == ACL_SUCCESS) {
        std::vector<uint16_t> output(EXAMPLE_SEQUENCE * EXAMPLE_HIDDEN);
        if (aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), allocations[MIXED_ALLOCATION_INDEX],
                        output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            result = 0;
            for (auto value : output)
                if ((value & 0x7fff) != 0)
                    result = 1;
        }
    }
    if (workspace)
        aclrtFree(workspace);
    for (auto* t : tensors)
        aclDestroyTensor(t);
    for (auto* p : allocations)
        aclrtFree(p);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::printf("TextGatedResidual example: %s\n", result == 0 ? "PASS" : "FAIL");
    return result;
}
