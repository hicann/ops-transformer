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
 * \file test_aclnn_stem_oam_prep_paged_kv.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_stem_oam_prep_paged_kv.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t batch = 1;
    int64_t totalBlocks = 8;
    int64_t numKvHeads = 4;
    int64_t dimQk = 128;
    int64_t maxKvBlocks = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t maxKb = 2;
    int64_t kflatDim = stemStride * dimQk;
    char *kvLayout = "BNBD";

    std::vector<int64_t> kCacheShape = {totalBlocks, numKvHeads, 64, dimQk};
    std::vector<int64_t> kvIndicesShape = {batch, maxKvBlocks};
    std::vector<int64_t> kScaleCacheShape = {totalBlocks, numKvHeads, 64, 1};
    std::vector<int64_t> vScaleShape = {numKvHeads};
    std::vector<int64_t> kFlatShape = {batch, numKvHeads, maxKb, kflatDim};
    std::vector<int64_t> vBiasShape = {batch, numKvHeads, maxKb};

    void *kCacheDeviceAddr = nullptr;
    void *vCacheDeviceAddr = nullptr;
    void *kvIndicesDeviceAddr = nullptr;
    void *kScaleCacheDeviceAddr = nullptr;
    void *vScaleDeviceAddr = nullptr;
    void *kFlatDeviceAddr = nullptr;
    void *vBiasDeviceAddr = nullptr;

    aclTensor *kCache = nullptr;
    aclTensor *vCache = nullptr;
    aclTensor *kvIndices = nullptr;
    aclIntArray *kvSeqLens = nullptr;
    aclTensor *kScaleCache = nullptr;
    aclTensor *vScale = nullptr;
    aclTensor *kFlat = nullptr;
    aclTensor *vBias = nullptr;

    std::vector<uint8_t> hostKCache(GetShapeSize(kCacheShape), 1);
    std::vector<uint8_t> hostVCache(GetShapeSize(kCacheShape), 1);
    std::vector<int32_t> hostKvIndices({0, 1, 2, 3, 4, 5, 6, 7});
    std::vector<int64_t> hostKvSeqLens({128});
    std::vector<float> hostKScaleCache(GetShapeSize(kScaleCacheShape), 1.0f);
    std::vector<float> hostVScale(numKvHeads, 1.0f);
    std::vector<uint16_t> hostKFlat(GetShapeSize(kFlatShape), 0);
    std::vector<float> hostVBias(GetShapeSize(vBiasShape), 0.0f);

    ret = CreateAclTensor(hostKCache, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kCache,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVCache, kCacheShape, &vCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &vCache,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKvIndices, kvIndicesShape, &kvIndicesDeviceAddr, aclDataType::ACL_INT32, &kvIndices,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    kvSeqLens = aclCreateIntArray(hostKvSeqLens.data(), hostKvSeqLens.size());
    ret = CreateAclTensor(hostKScaleCache, kScaleCacheShape, &kScaleCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &kScaleCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVScale, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKFlat, kFlatShape, &kFlatDeviceAddr, aclDataType::ACL_BF16, &kFlat,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVBias, vBiasShape, &vBiasDeviceAddr, aclDataType::ACL_FLOAT, &vBias,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnStemOamPrepPagedKvGetWorkspaceSize(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, 0.3,
                                                  kvLayout, stemBlockSize, stemStride, kFlat, vBias, &workspaceSize,
                                                  &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnStemOamPrepPagedKv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepPagedKv failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(kFlatShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), kFlatDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size && i < 10; i++) {
        float val;
        uint16_t bf16 = resultData[i];
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        std::memcpy(&val, &bits, sizeof(val));
        LOG_PRINT("result[%ld] is: %f\n", i, val);
    }

    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kvIndices);
    aclDestroyIntArray(kvSeqLens);
    aclDestroyTensor(kScaleCache);
    aclDestroyTensor(vScale);
    aclDestroyTensor(kFlat);
    aclDestroyTensor(vBias);

    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kvIndicesDeviceAddr);
    aclrtFree(kScaleCacheDeviceAddr);
    aclrtFree(vScaleDeviceAddr);
    aclrtFree(kFlatDeviceAddr);
    aclrtFree(vBiasDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
