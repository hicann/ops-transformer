/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_generic_block_sparse_attention_grad_metadata.cpp
 * \brief GenericBlockSparseAttentionGradMetadata 独立调用示例 (BNSD Layout)
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, static_cast<size_t>(size), hostData.data(), static_cast<size_t>(size),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

void PrintMetadataHeader(void **deviceAddr)
{
    std::vector<int64_t> header(8, 0);
    auto ret = aclrtMemcpy(header.data(), header.size() * sizeof(int64_t), *deviceAddr, header.size() * sizeof(int64_t),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy metadata header failed. ERROR: %d\n", ret); return);
    LOG_PRINT("metadata[0] total_num=%ld\n", header[0]);
    LOG_PRINT("metadata[1] total_block_cost=%ld\n", header[1]);
    LOG_PRINT("metadata[5] used_core_num=%ld\n", header[5]);
    LOG_PRINT("metadata[6] group_size=%ld\n", header[6]);
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t B = 1;
    const int64_t N1 = 1;
    const int64_t N2 = 1;
    const int64_t S1 = 128;
    const int64_t S2 = 128;
    const int64_t D = 128;
    const int64_t blockX = 1;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t maskType = 1;
    const int64_t isPackedGqa = 1;
    const int64_t softmaxPrecision = 0;
    const int64_t windowLeft = -1;
    const int64_t windowRight = -1;
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<int32_t> idxHost(static_cast<size_t>(GetShapeSize(idxShape)), -1);
    std::vector<int32_t> cntHost(static_cast<size_t>(GetShapeSize(cntShape)), 0);
    std::vector<int64_t> metaHost(static_cast<size_t>(metaSize), 0);

    for (int64_t q = 0; q < S1; ++q) {
        idxHost[static_cast<size_t>(q)] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *idxAddr = nullptr;
    void *cntAddr = nullptr;
    void *metaAddr = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *cnt = nullptr;
    aclTensor *metadata = nullptr;

    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT64, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockX, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

    char layout[] = "BNSD";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    LOG_PRINT("Calling aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize...\n");
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, isPackedGqa, layout, layout,
        maskType, softmaxPrecision, windowLeft, windowRight, metadata, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    LOG_PRINT("Calling aclnnGenericBlockSparseAttentionGradMetadata...\n");
    ret = aclnnGenericBlockSparseAttentionGradMetadata(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadata failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Metadata header:\n");
    PrintMetadataHeader(&metaAddr);

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("GenericBlockSparseAttentionGradMetadata test completed successfully.\n");
    return 0;
}
