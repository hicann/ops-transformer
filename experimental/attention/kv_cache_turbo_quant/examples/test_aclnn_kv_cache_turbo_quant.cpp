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
#include "aclnnop/aclnn_kv_cache_turbo_quant.h"

namespace {
constexpr int DEVICE_ID = 0;
constexpr int64_t TOKENS = 1;
constexpr int64_t HEADS = 4;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t QJL_DIM = 128;
constexpr int64_t MSE_BITS = 3;
constexpr int64_t OUTPUT_DTYPE = 0;

#define CHECK_ACL(expression) \
    do { \
        const aclError status = (expression); \
        if (status != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expression, static_cast<int>(status)); \
            return status; \
        } \
    } while (0)

struct TensorResource {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

int CreateTensor(const std::vector<int64_t>& shape, aclDataType dataType, size_t elementBytes, TensorResource& resource)
{
    const size_t bytes = static_cast<size_t>(Numel(shape)) * elementBytes;
    CHECK_ACL(aclrtMalloc(&resource.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(resource.device, bytes, 0, bytes));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), resource.device);
    return resource.tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

void DestroyTensor(TensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.device != nullptr) {
        aclrtFree(resource.device);
    }
}
} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(DEVICE_ID));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    TensorResource kvVectors;
    TensorResource rotationMatrix;
    TensorResource qjlMatrix;
    TensorResource quantIndex;
    TensorResource quantQjl;
    TensorResource quantNorm;
    TensorResource quantGamma;
    CHECK_ACL(CreateTensor({TOKENS, HEADS, HEAD_DIM}, ACL_BF16, sizeof(uint16_t), kvVectors));
    CHECK_ACL(CreateTensor({HEAD_DIM, HEAD_DIM}, ACL_FLOAT, sizeof(float), rotationMatrix));
    CHECK_ACL(CreateTensor({QJL_DIM, HEAD_DIM}, ACL_FLOAT, sizeof(float), qjlMatrix));
    CHECK_ACL(CreateTensor({TOKENS, HEADS, (HEAD_DIM * MSE_BITS + 7) / 8}, ACL_UINT8, sizeof(uint8_t), quantIndex));
    CHECK_ACL(CreateTensor({TOKENS, HEADS, (QJL_DIM + 7) / 8}, ACL_UINT8, sizeof(uint8_t), quantQjl));
    CHECK_ACL(CreateTensor({TOKENS, HEADS}, ACL_BF16, sizeof(uint16_t), quantNorm));
    CHECK_ACL(CreateTensor({TOKENS, HEADS}, ACL_BF16, sizeof(uint16_t), quantGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnKvCacheTurboQuantGetWorkspaceSize(kvVectors.tensor, rotationMatrix.tensor, qjlMatrix.tensor,
                                                     MSE_BITS, OUTPUT_DTYPE, quantIndex.tensor, quantQjl.tensor,
                                                     quantNorm.tensor, quantGamma.tensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> normResult(TOKENS * HEADS, 1U);
    std::vector<uint16_t> gammaResult(TOKENS * HEADS, 1U);
    CHECK_ACL(aclrtMemcpy(normResult.data(), normResult.size() * sizeof(uint16_t), quantNorm.device,
                          normResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(gammaResult.data(), gammaResult.size() * sizeof(uint16_t), quantGamma.device,
                          gammaResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    bool passed = true;
    for (uint16_t value : normResult) {
        passed = passed && value == 0U;
    }
    for (uint16_t value : gammaResult) {
        passed = passed && value == 0U;
    }
    std::printf("KvCacheTurboQuant example: %s, workspace=%lu bytes\n", passed ? "PASSED" : "FAILED", workspaceSize);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    DestroyTensor(kvVectors);
    DestroyTensor(rotationMatrix);
    DestroyTensor(qjlMatrix);
    DestroyTensor(quantIndex);
    DestroyTensor(quantQjl);
    DestroyTensor(quantNorm);
    DestroyTensor(quantGamma);
    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return passed ? 0 : 1;
}
