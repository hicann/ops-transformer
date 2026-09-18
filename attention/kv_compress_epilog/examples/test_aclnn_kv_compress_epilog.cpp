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
#include "aclnn_kv_compress_epilog.h"

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
bool Check(aclError status, const char *operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    LOG_PRINT("%s failed: %d\n", operation, status);
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
            LOG_PRINT("aclCreateTensor failed\n");
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

    // Mode 1, d=256: 128 rope bytes + 192 nope bytes + 3 scale bytes = 323 bytes per slot.
    // The cache row has 384 bytes; each token writes a distinct slot within the 2048-slot cache.
    constexpr int64_t blockNum = 128;
    constexpr int64_t blockSize = 16;
    constexpr int64_t headDim = 384;
    constexpr int64_t tokenNum = 1024;
    constexpr int64_t d = 256;
    std::vector<uint8_t> cacheData(blockNum * blockSize * headDim, 0);
    // BF16 bit pattern for 1.0.
    std::vector<uint16_t> xData(tokenNum * d, 0x3f80);
    std::vector<int32_t> slotData(tokenNum);
    for (int32_t i = 0; i < tokenNum; ++i) {
        slotData[i] = i;
    }
    auto cache =
        resources.CreateTensor({blockNum, blockSize, 1, headDim}, ACL_UINT8, cacheData.data(), cacheData.size());
    auto x = resources.CreateTensor({tokenNum, d}, ACL_BF16, xData.data(), xData.size() * sizeof(uint16_t));
    auto slotMapping =
        resources.CreateTensor({tokenNum}, ACL_INT32, slotData.data(), slotData.size() * sizeof(int32_t));
    if (cache == nullptr || x == nullptr || slotMapping == nullptr) {
        return 1;
    }

    int64_t quantGroupSize = 64;
    int64_t quantMode = 1;
    bool roundScale = true;
    float xScale = 1.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (!Check(aclnnKvCompressEpilogGetWorkspaceSize(cache, x, slotMapping, quantGroupSize, quantMode, roundScale,
                                                     xScale, &workspaceSize, &executor),
               "aclnnKvCompressEpilogGetWorkspaceSize")) {
        return 1;
    }
    if (workspaceSize != 0 &&
        !Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc workspace")) {
        return 1;
    }
    if (!Check(aclnnKvCompressEpilog(resources.workspace, workspaceSize, executor, resources.stream),
               "aclnnKvCompressEpilog") ||
        !Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }

    LOG_PRINT("KvCompressEpilog execution completed successfully!\n");
    return 0;
}
