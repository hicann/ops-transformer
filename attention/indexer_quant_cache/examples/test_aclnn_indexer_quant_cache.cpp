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
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_indexer_quant_cache.h"

namespace {
bool Check(aclError status, const char *operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << status << std::endl;
    return false;
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    void *workspace = nullptr;
    std::vector<void *> buffers;
    std::vector<aclTensor *> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream (cleanup)");
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto buffer : buffers) {
            Check(aclrtFree(buffer), "aclrtFree");
        }
        if (workspace != nullptr) {
            Check(aclrtFree(workspace), "aclrtFree workspace");
        }
        if (stream != nullptr) {
            Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        }
        if (deviceSet) {
            Check(aclrtResetDevice(0), "aclrtResetDevice");
        }
        if (initialized) {
            Check(aclFinalize(), "aclFinalize");
        }
    }

    aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, const void *data, size_t bytes)
    {
        void *deviceData = nullptr;
        if (!Check(aclrtMalloc(&deviceData, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc tensor")) {
            return nullptr;
        }
        buffers.push_back(deviceData);
        if (!Check(aclrtMemcpy(deviceData, bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D")) {
            return nullptr;
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), deviceData);
        if (tensor == nullptr) {
            std::cerr << "aclCreateTensor failed" << std::endl;
            return nullptr;
        }
        tensors.push_back(tensor);
        return tensor;
    }
};
} // namespace

int main()
{
    Resources resources;
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!Check(aclrtSetDevice(0), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    constexpr int64_t slots = 32;
    constexpr int64_t d = 32;
    constexpr int64_t tokens = 2;
    std::vector<uint8_t> cacheData(slots * d, 0);
    std::vector<float> scaleData(slots, 2.0f);
    // IEEE FP16 bit pattern for 1.0. Token 1 is skipped; only slot 1 is updated.
    std::vector<uint16_t> xData(tokens * d, 0x3c00);
    std::vector<int32_t> slotData = {1, -1};
    auto cache = resources.CreateTensor({4, 8, 1, d}, ACL_FLOAT8_E4M3FN, cacheData.data(), cacheData.size());
    auto scale = resources.CreateTensor({4, 8, 1, 1}, ACL_FLOAT, scaleData.data(), scaleData.size() * sizeof(float));
    auto x = resources.CreateTensor({tokens, d}, ACL_FLOAT16, xData.data(), xData.size() * sizeof(uint16_t));
    auto mapping = resources.CreateTensor({tokens}, ACL_INT32, slotData.data(), slotData.size() * sizeof(int32_t));
    if (cache == nullptr || scale == nullptr || x == nullptr || mapping == nullptr) {
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (!Check(
            aclnnIndexerQuantCacheGetWorkspaceSize(cache, scale, x, mapping, 1, true, 1.0f, &workspaceSize, &executor),
            "GetWorkspaceSize")) {
        return 1;
    }
    if (workspaceSize != 0 &&
        !Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc workspace")) {
        return 1;
    }
    if (!Check(aclnnIndexerQuantCache(resources.workspace, workspaceSize, executor, resources.stream),
               "aclnnIndexerQuantCache") ||
        !Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }
    if (!Check(aclrtMemcpy(cacheData.data(), cacheData.size(), resources.buffers[0], cacheData.size(),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy cache D2H") ||
        !Check(aclrtMemcpy(scaleData.data(), scaleData.size() * sizeof(float), resources.buffers[1],
                           scaleData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy scale D2H")) {
        return 1;
    }

    // Normal E4M3: ceil_pow2(1/448) = 1/256; 1/(1/256) = 256, encoded as 0x78.
    // Verify both complete outputs, including every untouched slot.
    for (int64_t slot = 0; slot < slots; ++slot) {
        const float expectedScale = slot == 1 ? 1.0f / 256.0f : 2.0f;
        if (scaleData[slot] != expectedScale) {
            std::cerr << "scale mismatch at slot " << slot << std::endl;
            return 1;
        }
        for (int64_t col = 0; col < d; ++col) {
            const uint8_t expected = slot == 1 ? 0x78 : 0;
            if (cacheData[slot * d + col] != expected) {
                std::cerr << "cache mismatch at slot " << slot << ", column " << col << std::endl;
                return 1;
            }
        }
    }
    std::cout << "IndexerQuantCache PASS: cache, scale and untouched slots verified." << std::endl;
    return 0;
}
