/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v3_metadata.h"

#define CHECK_RET(expr) \
    do { \
        int ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expr, ret); \
            return ret; \
        } \
    } while (0)

void CheckCleanup(int cleanupRet, const char* operation, int& ret)
{
    if (cleanupRet != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", operation, cleanupRet);
        if (ret == ACL_SUCCESS) {
            ret = cleanupRet;
        }
    }
}

struct Resources {
    aclrtStream stream;
    int& cleanupRet;
    std::vector<aclTensor*> tensors;
    std::vector<void*> allocations;

    Resources(aclrtStream inputStream, int& cleanupStatus)
        : stream(inputStream),
          cleanupRet(cleanupStatus)
    {}
    ~Resources()
    {
        // 异常返回时也先等待已提交任务，随后释放Tensor描述和Device内存。
        CheckCleanup(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", cleanupRet);
        for (auto tensor : tensors) {
            CheckCleanup(aclDestroyTensor(tensor), "aclDestroyTensor", cleanupRet);
        }
        for (auto address : allocations) {
            CheckCleanup(aclrtFree(address), "aclrtFree", cleanupRet);
        }
    }

    int Allocate(uint64_t bytes, void** address)
    {
        CHECK_RET(aclrtMalloc(address, std::max<uint64_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST));
        allocations.push_back(*address);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype,
                     aclTensor** tensor, void** address)
    {
        uint64_t bytes = host.size() * sizeof(T);
        CHECK_RET(Allocate(bytes, address));
        if (bytes != 0) {
            CHECK_RET(aclrtMemcpy(*address, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                  shape.size(), *address);
        if (*tensor == nullptr) {
            std::printf("aclCreateTensor failed\n");
            return 1;
        }
        tensors.push_back(*tensor);
        return ACL_SUCCESS;
    }
};

int RunExample(aclrtStream stream, int& cleanupRet)
{
    Resources resources(stream, cleanupRet);
    constexpr int64_t batch = 2;
    constexpr int64_t queryHeads = 32;
    constexpr int64_t keyHeads = 1;
    constexpr int64_t headDim = 128;
    constexpr int64_t keyLength = 2048;
    constexpr int64_t topk = 2048;
    constexpr int64_t candidateCapacity = 2048;
    constexpr int64_t candidateBlockSize = 8;
    char layoutQ[] = "TND";
    char layoutK[] = "PA_BBND";

    // 每个batch一个Query，Key有效长度均为2048；metadata使用固定1024个INT32元素。
    aclTensor* cuQ = nullptr;
    aclTensor* usedK = nullptr;
    aclTensor* metadata = nullptr;
    void* cuQAddress = nullptr;
    void* usedKAddress = nullptr;
    void* metadataAddress = nullptr;
    CHECK_RET(resources.CreateTensor<int32_t>({0, 1, 2}, {batch + 1}, ACL_INT32, &cuQ, &cuQAddress));
    CHECK_RET(resources.CreateTensor<int32_t>({keyLength, keyLength}, {batch}, ACL_INT32, &usedK, &usedKAddress));
    CHECK_RET(
        resources.CreateTensor<int32_t>(std::vector<int32_t>(1024), {1024}, ACL_INT32, &metadata, &metadataAddress));

    // 先生成metadata；后续计算使用完全相同的序列信息及属性。
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor* metadataExecutor = nullptr;
    CHECK_RET(aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
        cuQ, nullptr, nullptr, usedK, nullptr, nullptr, queryHeads, keyHeads, headDim, topk, 2, batch, 1, keyLength,
        layoutQ, layoutK, 0, 1, candidateCapacity, candidateBlockSize, nullptr, metadata, &metadataWorkspaceSize,
        &metadataExecutor));
    void* metadataWorkspace = nullptr;
    if (metadataWorkspaceSize != 0) {
        CHECK_RET(resources.Allocate(metadataWorkspaceSize, &metadataWorkspace));
    }
    CHECK_RET(aclnnQuantLightningIndexerV3Metadata(metadataWorkspace, metadataWorkspaceSize, metadataExecutor, stream));

    CHECK_RET(aclrtSynchronizeStream(stream));
    std::printf("metadata generated: INT32[1024]\n");
    return ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        CheckCleanup(aclFinalize(), "aclFinalize", ret);
        return ret;
    }
    int cleanupRet = ACL_SUCCESS;
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret == ACL_SUCCESS) {
        ret = RunExample(stream, cleanupRet);
        CheckCleanup(aclrtDestroyStream(stream), "aclrtDestroyStream", cleanupRet);
    }
    CheckCleanup(aclrtResetDevice(deviceId), "aclrtResetDevice", cleanupRet);
    CheckCleanup(aclFinalize(), "aclFinalize", cleanupRet);
    return ret != ACL_SUCCESS ? ret : cleanupRet;
}
