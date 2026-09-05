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
 * \file test_aclnn_generic_block_sparse_attention_grad.cpp
 * \brief GenericBlockSparseAttentionGrad + Metadata 调用示例 (BNSD Layout)
 *
 * 流程：先调用 aclnnGenericBlockSparseAttentionGradMetadata 生成分核 metadata，
 * 再调用 aclnnGenericBlockSparseAttentionGrad 计算 dQ/dK/dV。
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad.h"
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

void PrintOutResult(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(static_cast<size_t>(size), 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           static_cast<size_t>(size) * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 8; ++i) {
        LOG_PRINT("result[%ld] raw_fp16_bits=0x%04x\n", i, static_cast<unsigned>(resultData[i]));
    }
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

} // namespace

int main()
{
    // 1. device/context/stream 初始化
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. BNSD smoke 参数（对齐 GSAG 约束：D=128, blockShape=[1,128]）
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
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(D));
    // metadata elems = TASK_LIST_OFFSET(80) + B * N1 * J * TASK_ENTRY_SIZE(4)
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> qShape = {B, N1, S1, D};
    std::vector<int64_t> kvShape = {B, N2, S2, D};
    std::vector<int64_t> lseShape = {B, N1, S1};
    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    // 3. Host 数据：小常数输入 + 单 KV 块全选 Q
    std::vector<uint16_t> qHost(static_cast<size_t>(GetShapeSize(qShape)), 0x2E66); // ~0.1
    std::vector<uint16_t> kHost(static_cast<size_t>(GetShapeSize(kvShape)), 0x2E66);
    std::vector<uint16_t> vHost(static_cast<size_t>(GetShapeSize(kvShape)), 0x2E66);
    std::vector<uint16_t> doutHost(static_cast<size_t>(GetShapeSize(qShape)), 0x211E); // ~0.01
    std::vector<uint16_t> outHost(static_cast<size_t>(GetShapeSize(qShape)), 0x2E66);
    std::vector<float> lseHost(static_cast<size_t>(GetShapeSize(lseShape)), 5.0f);
    std::vector<int32_t> idxHost(static_cast<size_t>(GetShapeSize(idxShape)), -1);
    std::vector<int32_t> cntHost(static_cast<size_t>(GetShapeSize(cntShape)), 0);
    std::vector<int64_t> metaHost(static_cast<size_t>(metaSize), 0);
    std::vector<uint16_t> dqHost(static_cast<size_t>(GetShapeSize(qShape)), 0);
    std::vector<uint16_t> dkHost(static_cast<size_t>(GetShapeSize(kvShape)), 0);
    std::vector<uint16_t> dvHost(static_cast<size_t>(GetShapeSize(kvShape)), 0);

    for (int64_t q = 0; q < S1; ++q) {
        idxHost[static_cast<size_t>(q)] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *vAddr = nullptr;
    void *doutAddr = nullptr;
    void *outAddr = nullptr;
    void *lseAddr = nullptr;
    void *idxAddr = nullptr;
    void *cntAddr = nullptr;
    void *metaAddr = nullptr;
    void *dqAddr = nullptr;
    void *dkAddr = nullptr;
    void *dvAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *dout = nullptr;
    aclTensor *out = nullptr;
    aclTensor *lse = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *cnt = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dv = nullptr;

    ret = CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHost, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHost, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, qShape, &outAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lseHost, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT64, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHost, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHost, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dvHost, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockX, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

    char qLayout[] = "BNSD";
    char kvLayout[] = "BNSD";

    // 4. 先跑 Metadata
    uint64_t metaWsSize = 0;
    aclOpExecutor *metaExecutor = nullptr;
    LOG_PRINT("Calling aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize...\n");
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, isPackedGqa, qLayout, kvLayout,
        maskType, softmaxPrecision, windowLeft, windowRight, metadata, &metaWsSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metaWs = nullptr;
    if (metaWsSize > 0) {
        ret = aclrtMalloc(&metaWs, metaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(metaWs, metaWsSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadata failed. ERROR: %d\n", ret);
              return ret);

    // 5. 再跑 Grad
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    LOG_PRINT("Calling aclnnGenericBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
        q, k, v, dout, out, lse, idx, cnt, metadata, nullptr /*attenMask*/, nullptr /*cuQ*/, nullptr /*cuKv*/,
        nullptr /*sequsedQ*/, nullptr /*sequsedKv*/, blockShape, isPackedGqa, qLayout, kvLayout, scaleValue, maskType,
        softmaxPrecision, windowLeft, windowRight, dq, dk, dv, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", static_cast<unsigned long>(workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    LOG_PRINT("Calling aclnnGenericBlockSparseAttentionGrad...\n");
    ret = aclnnGenericBlockSparseAttentionGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGrad failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("dQuery sample:\n");
    PrintOutResult(qShape, &dqAddr);

    // 6. 资源释放
    aclDestroyIntArray(blockShape);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(dout);
    aclDestroyTensor(out);
    aclDestroyTensor(lse);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dv);

    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(vAddr);
    aclrtFree(doutAddr);
    aclrtFree(outAddr);
    aclrtFree(lseAddr);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    aclrtFree(dqAddr);
    aclrtFree(dkAddr);
    aclrtFree(dvAddr);
    if (metaWsSize > 0) {
        aclrtFree(metaWs);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("GenericBlockSparseAttentionGrad test completed successfully.\n");
    return 0;
}
