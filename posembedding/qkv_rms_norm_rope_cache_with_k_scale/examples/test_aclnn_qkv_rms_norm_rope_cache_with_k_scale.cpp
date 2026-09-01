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
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

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

struct TensorResource {
    aclTensor *tensor = nullptr;
    void *deviceAddr = nullptr;
};

struct AclResource {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool aclInited = false;
    bool deviceSet = false;
    std::vector<TensorResource *> tensors;
    aclIntArray *headNums = nullptr;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

int Init(int32_t deviceId, AclResource &resource)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    resource.aclInited = true;

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    resource.deviceId = deviceId;
    resource.deviceSet = true;

    ret = aclrtCreateStream(&resource.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void FreeResource(AclResource &resource)
{
    for (auto *tensorResource : resource.tensors) {
        if (tensorResource != nullptr && tensorResource->tensor != nullptr) {
            aclDestroyTensor(tensorResource->tensor);
            tensorResource->tensor = nullptr;
        }
    }
    if (resource.headNums != nullptr) {
        aclDestroyIntArray(resource.headNums);
        resource.headNums = nullptr;
    }
    for (auto *tensorResource : resource.tensors) {
        if (tensorResource != nullptr && tensorResource->deviceAddr != nullptr) {
            aclrtFree(tensorResource->deviceAddr);
            tensorResource->deviceAddr = nullptr;
        }
    }
    if (resource.workspaceAddr != nullptr) {
        aclrtFree(resource.workspaceAddr);
        resource.workspaceAddr = nullptr;
    }
    if (resource.stream != nullptr) {
        aclrtDestroyStream(resource.stream);
        resource.stream = nullptr;
    }
    if (resource.deviceSet) {
        aclrtResetDevice(resource.deviceId);
        resource.deviceSet = false;
    }
    if (resource.aclInited) {
        aclFinalize();
        resource.aclInited = false;
    }
}

int ReturnAfterCleanup(int ret, AclResource &resource)
{
    FreeResource(resource);
    return ret;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                    TensorResource &resource)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(&resource.deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resource.deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides = GetContiguousStrides(shape);
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), resource.deviceAddr);
    CHECK_RET(resource.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

int main()
{
    AclResource resource;
    auto ret = Init(0, resource);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
              return ReturnAfterCleanup(ret, resource));

    constexpr int64_t T = 4;
    constexpr int64_t Nq = 16;
    constexpr int64_t Nk = 2;
    constexpr int64_t Nv = 2;
    constexpr int64_t D = 128;
    constexpr int64_t Batch = 1;
    constexpr int64_t MaxSeqLen = 16;
    constexpr int64_t BlockNum = 1;
    constexpr int64_t BlockSize = 16;

    std::vector<int64_t> qkvShape = {T, Nq + Nk + Nv, D};
    std::vector<int64_t> qGammaShape = {D};
    std::vector<int64_t> kGammaShape = {D};
    std::vector<int64_t> cosSinShape = {MaxSeqLen, D};
    std::vector<int64_t> slotMappingShape = {T};
    std::vector<int64_t> kCacheShape = {BlockNum, Nk, BlockSize, D};
    std::vector<int64_t> vCacheShape = {BlockNum, Nv, BlockSize, D};
    std::vector<int64_t> kScaleCacheShape = {BlockNum, Nk, BlockSize, 1};
    std::vector<int64_t> queryStartLocShape = {Batch + 1};
    std::vector<int64_t> seqLensShape = {Batch};
    std::vector<int64_t> rotationShape = {D, D};
    std::vector<int64_t> vScaleShape = {Nv};
    std::vector<int64_t> qOutShape = {T, Nq, D};
    std::vector<int64_t> qScaleShape = {T, Nq};

    std::vector<uint16_t> qkvHostData(GetShapeSize(qkvShape), FloatToBf16(0.125f));
    std::vector<float> qGammaHostData(GetShapeSize(qGammaShape), 1.0f);
    std::vector<float> kGammaHostData(GetShapeSize(kGammaShape), 1.0f);
    std::vector<float> cosSinHostData(GetShapeSize(cosSinShape), 0.0f);
    for (int64_t row = 0; row < MaxSeqLen; ++row) {
        for (int64_t col = 0; col < D / 2; ++col) {
            cosSinHostData[row * D + col] = 1.0f;
        }
    }
    std::vector<int32_t> slotMappingHostData = {0, 1, 2, 3};
    std::vector<uint8_t> kCacheHostData(GetShapeSize(kCacheShape), 0);
    std::vector<uint8_t> vCacheHostData(GetShapeSize(vCacheShape), 0);
    std::vector<float> kScaleCacheHostData(GetShapeSize(kScaleCacheShape), 0.0f);
    std::vector<int32_t> queryStartLocHostData = {0, T};
    std::vector<int32_t> seqLensHostData = {T};
    std::vector<uint16_t> rotationHostData(GetShapeSize(rotationShape), FloatToBf16(0.0f));
    for (int64_t i = 0; i < D; ++i) {
        rotationHostData[i * D + i] = FloatToBf16(1.0f);
    }
    std::vector<float> vScaleHostData(GetShapeSize(vScaleShape), 1.0f);
    std::vector<uint8_t> qOutHostData(GetShapeSize(qOutShape), 0);
    std::vector<float> qScaleHostData(GetShapeSize(qScaleShape), 0.0f);

    TensorResource qkv;
    TensorResource qGamma;
    TensorResource kGamma;
    TensorResource cosSin;
    TensorResource slotMapping;
    TensorResource kCache;
    TensorResource vCache;
    TensorResource kScaleCache;
    TensorResource queryStartLoc;
    TensorResource seqLens;
    TensorResource rotation;
    TensorResource vScale;
    TensorResource qOut;
    TensorResource qScale;
    resource.tensors = {&qkv,         &qGamma,        &kGamma,  &cosSin,   &slotMapping, &kCache, &vCache,
                        &kScaleCache, &queryStartLoc, &seqLens, &rotation, &vScale,      &qOut,   &qScale};

    ret = CreateAclTensor(qkvHostData, qkvShape, ACL_BF16, qkv);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(qGammaHostData, qGammaShape, ACL_FLOAT, qGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(kGammaHostData, kGammaShape, ACL_FLOAT, kGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(cosSinHostData, cosSinShape, ACL_FLOAT, cosSin);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(slotMappingHostData, slotMappingShape, ACL_INT32, slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(kCacheHostData, kCacheShape, ACL_FLOAT8_E4M3FN, kCache);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(vCacheHostData, vCacheShape, ACL_FLOAT8_E4M3FN, vCache);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(kScaleCacheHostData, kScaleCacheShape, ACL_FLOAT, kScaleCache);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, ACL_INT32, queryStartLoc);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(seqLensHostData, seqLensShape, ACL_INT32, seqLens);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(rotationHostData, rotationShape, ACL_BF16, rotation);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(vScaleHostData, vScaleShape, ACL_FLOAT, vScale);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(qOutHostData, qOutShape, ACL_FLOAT8_E4M3FN, qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
    ret = CreateAclTensor(qScaleHostData, qScaleShape, ACL_FLOAT, qScale);
    CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));

    std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
    resource.headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());
    CHECK_RET(resource.headNums != nullptr, LOG_PRINT("aclCreateIntArray failed.\n");
              return ReturnAfterCleanup(ACL_ERROR_INVALID_PARAM, resource));

    const char *layoutQkv = "TND";
    const char *layoutQOut = "TND";
    float epsilon = 1e-6f;
    uint64_t workspaceSize = 0;
    aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
        qkv.tensor, qGamma.tensor, kGamma.tensor, cosSin.tensor, slotMapping.tensor, kCache.tensor, vCache.tensor,
        kScaleCache.tensor, queryStartLoc.tensor, seqLens.tensor, rotation.tensor, vScale.tensor, nullptr,
        resource.headNums, layoutQkv, layoutQOut, epsilon, nullptr, "PerTokenPerHead", "PerTokenPerHead", qOut.tensor,
        qScale.tensor, &workspaceSize, &resource.executor);
    CHECK_RET(status == ACL_SUCCESS,
              LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", status);
              return ReturnAfterCleanup(static_cast<int>(status), resource));

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&resource.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  return ReturnAfterCleanup(ret, resource));
    }

    status =
        aclnnQkvRmsNormRopeCacheWithKScale(resource.workspaceAddr, workspaceSize, resource.executor, resource.stream);
    CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale failed. ERROR: %d\n", status);
              return ReturnAfterCleanup(static_cast<int>(status), resource));

    ret = aclrtSynchronizeStream(resource.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              return ReturnAfterCleanup(ret, resource));
    LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale execute success.\n");
    FreeResource(resource);
    return 0;
}
