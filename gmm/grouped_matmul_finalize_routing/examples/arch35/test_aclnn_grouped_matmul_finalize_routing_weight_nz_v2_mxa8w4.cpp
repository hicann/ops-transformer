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
 * \file test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4.cpp
 * \brief MxA8W4数据流示例程序
 *
 * MxA8W4数据流说明：
 * - x1: FLOAT8_E4M3FN, ND格式, shape=(m, k)
 * - x2: FLOAT4_E2M1, NZ_C0_32格式, viewShape=(e, n, k), transposeX2=true
 * - scale: FLOAT8_E8M0, ND格式, shape=(e, n, ceil(k/64), 2)
 * - pertokenScale: FLOAT8_E8M0, ND格式, shape=(m, ceil(k/64), 2)
 *
 * 约束条件：
 * - k % 32 == 0
 * - n % 32 == 0
 * - e <= 1024
 * - transposeX2固定为true
 * - offsetOptional必须为nullptr
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "aclnnop/aclnn_npu_format_cast.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
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

int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0)
        return 0;
    if (a <= 0)
        return 0;
    return (a - 1) / b + 1;
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t *storageShape,
                              uint64_t storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                              aclFormat format)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape,
                              storageShapeSize, *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor, aclrtStream &stream)
{
    void *srcDeviceAddr = nullptr;
    aclTensor *srcTensor = nullptr;
    auto size = hostData.size() * sizeof(T);

    auto ret = aclrtMalloc(&srcDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(srcDeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    srcTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), srcDeviceAddr);

    int64_t *dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    // aclnnNpuFormatCast private format code: 29 means FRACTAL_NZ.
    constexpr int64_t kNpuFormatFractalNz = 29;
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, kNpuFormatFractalNz, aclFormat::ACL_FORMAT_ND, &dstShape,
                                                   &dstShapeSize, &actualFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);

    aclTensor *dstTensor = nullptr;
    void *dstDeviceAddr = nullptr;

    uint64_t tensorSize = 1;
    for (uint64_t i = 0; i < dstShapeSize; i++) {
        tensorSize *= dstShape[i];
    }
    std::vector<T> dstTensorHostData(tensorSize, 0);

    ret = CreateAclTensorWithFormat(dstTensorHostData, shape, dstShape, dstShapeSize, &dstDeviceAddr, dataType,
                                    &dstTensor, static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    *tensor = dstTensor;
    *deviceAddr = dstDeviceAddr;

    aclDestroyTensor(srcTensor);
    aclrtFree(srcDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return ACL_SUCCESS;
}

int RunExample(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t batch = 24;
    int64_t bsdp = 8;
    int64_t dtype = 0;
    float shareInputWeight = 1.0;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = true;
    int64_t groupListType = 1;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> wShape = {e, n, k};
    std::vector<int64_t> scaleShape = {e, n, ceil_k_64, 2};
    std::vector<int64_t> pertokenScaleShape = {m, ceil_k_64, 2};
    std::vector<int64_t> groupListShape = {e};
    std::vector<int64_t> sharedInputShape = {bsdp, n};
    std::vector<int64_t> logitShape = {m};
    std::vector<int64_t> rowIndexShape = {m};
    std::vector<int64_t> outShape = {batch, n};

    void *xDeviceAddr = nullptr;
    void *wDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *sharedInputDeviceAddr = nullptr;
    void *logitDeviceAddr = nullptr;
    void *rowIndexDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *w = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *pertokenScale = nullptr;
    aclTensor *sharedInput = nullptr;
    aclTensor *logit = nullptr;
    aclTensor *rowIndex = nullptr;
    aclTensor *out = nullptr;

    std::vector<uint8_t> xHostData(GetShapeSize(xShape));
    std::vector<uint8_t> wHostData(GetShapeSize(wShape));
    std::vector<uint8_t> scaleHostData(GetShapeSize(scaleShape));
    std::vector<uint8_t> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
    std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape), m / e);

    std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
    std::vector<float> logitHostData(GetShapeSize(logitShape));
    std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
    std::vector<float> outHostData(GetShapeSize(outShape));

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

    ret = CreateAclTensorNz(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &w, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);

    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);

    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                          aclDestroyTensor);

    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);

    ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                          &sharedInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);

    ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);

    ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
        x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
        shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < std::min((int64_t)10, size); i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = RunExample(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunExample failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
