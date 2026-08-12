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
 * \file test_aclnn_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_quant_sparse_flash_mla_metadata.h"

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

uint16_t FloatToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t roundingBias = 0x7fffu + lsb;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void PrintOutResult(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
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
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

std::vector<uint16_t> MakeBf16Data(int64_t size, float value)
{
    std::vector<uint16_t> data(static_cast<size_t>(size), FloatToBf16(value));
    return data;
}

std::vector<uint8_t> MakeHif8Data(int64_t size)
{
    std::vector<uint8_t> data(static_cast<size_t>(size), 0x08);
    return data;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t B = 1;
    int64_t S1 = 1;
    int64_t S2 = 1024;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 512;
    int64_t K = 512;
    int64_t oriBlockSize = 128;
    int64_t cmpBlockSize = 128;
    int64_t s2Act = 1024;
    int64_t cmpRatio = 4;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
    int64_t oriMaskMode = 4;
    int64_t cmpMaskMode = 3;
    int64_t quantMode = 1;
    double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

    int64_t kvD = D;

    int64_t T1 = B * S1;
    int64_t cmpKvLen = s2Act / cmpRatio;
    int64_t oriBlockNum = ((s2Act + oriBlockSize - 1) / oriBlockSize) * B;
    int64_t cmpBlockNum = ((cmpKvLen + cmpBlockSize - 1) / cmpBlockSize) * B;

    std::vector<int64_t> qShape = {T1, N1, D};
    std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, kvD};
    std::vector<int64_t> cmpKvShape = {cmpBlockNum, cmpBlockSize, N2, kvD};
    std::vector<int64_t> cmpSparseIndicesShape = {T1, N2, K};
    std::vector<int64_t> oriBlockTableShape = {B, (s2Act + oriBlockSize - 1) / oriBlockSize};
    std::vector<int64_t> cmpBlockTableShape = {B, (cmpKvLen + cmpBlockSize - 1) / cmpBlockSize};
    std::vector<int64_t> cuSeqLensQShape = {B + 1};
    std::vector<int64_t> seqUsedOriKvShape = {B};
    std::vector<int64_t> seqUsedCmpKvShape = {B};
    std::vector<int64_t> cmpResidualKvShape = {B};
    std::vector<int64_t> sinksShape = {N1};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> descaleShape = {1};
    std::vector<int64_t> attnOutShape = {T1, N1, D};
    std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
    std::vector<int64_t> emptyShape = {0};

    void *qDeviceAddr = nullptr;
    void *oriKvDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;
    void *qDescaleDeviceAddr = nullptr;
    void *oriKvDescaleDeviceAddr = nullptr;
    void *cmpKvDescaleDeviceAddr = nullptr;
    void *cmpSparseIndicesDeviceAddr = nullptr;
    void *oriBlockTableDeviceAddr = nullptr;
    void *cmpBlockTableDeviceAddr = nullptr;
    void *cuSeqLensQDeviceAddr = nullptr;
    void *cuSeqLensOriKvDeviceAddr = nullptr;
    void *cuSeqLensCmpKvDeviceAddr = nullptr;
    void *seqUsedQDeviceAddr = nullptr;
    void *seqUsedOriKvDeviceAddr = nullptr;
    void *seqUsedCmpKvDeviceAddr = nullptr;
    void *cmpResidualKvDeviceAddr = nullptr;
    void *sinksDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *oriKv = nullptr;
    aclTensor *cmpKv = nullptr;
    aclTensor *qDescale = nullptr;
    aclTensor *oriKvDescale = nullptr;
    aclTensor *cmpKvDescale = nullptr;
    aclTensor *cmpSparseIndices = nullptr;
    aclTensor *oriBlockTable = nullptr;
    aclTensor *cmpBlockTable = nullptr;
    aclTensor *cuSeqLensQ = nullptr;
    aclTensor *cuSeqLensOriKv = nullptr;
    aclTensor *cuSeqLensCmpKv = nullptr;
    aclTensor *seqUsedQ = nullptr;
    aclTensor *seqUsedOriKv = nullptr;
    aclTensor *seqUsedCmpKv = nullptr;
    aclTensor *cmpResidualKv = nullptr;
    aclTensor *sinks = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *softmaxLse = nullptr;

    int64_t qSize = GetShapeSize(qShape);
    int64_t oriKvSize = GetShapeSize(oriKvShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);
    int64_t cmpSparseIndicesSize = GetShapeSize(cmpSparseIndicesShape);
    int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
    int64_t cmpBlockTableSize = GetShapeSize(cmpBlockTableShape);
    int64_t attnOutSize = GetShapeSize(attnOutShape);
    int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

    std::vector<uint8_t> qHostData = MakeHif8Data(qSize);
    std::vector<uint8_t> oriKvHostData = MakeHif8Data(oriKvSize);
    std::vector<uint8_t> cmpKvHostData = MakeHif8Data(cmpKvSize);
    std::vector<float> qDescaleHostData = {1.0f};
    std::vector<float> oriKvDescaleHostData = {1.0f};
    std::vector<float> cmpKvDescaleHostData = {1.0f};
    std::vector<int32_t> cmpSparseIndicesHostData(cmpSparseIndicesSize);
    std::vector<int32_t> oriBlockTableHostData(oriBlockTableSize);
    std::iota(oriBlockTableHostData.begin(), oriBlockTableHostData.end(), 0);
    std::vector<int32_t> cmpBlockTableHostData(cmpBlockTableSize);
    std::iota(cmpBlockTableHostData.begin(), cmpBlockTableHostData.end(), 0);
    std::vector<int32_t> cuSeqLensQHostData(B + 1);
    for (int64_t i = 0; i <= B; i++) {
        cuSeqLensQHostData[i] = static_cast<int32_t>(i * S1);
    }
    std::vector<int32_t> emptyHostData;
    std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(s2Act));
    std::vector<int32_t> seqUsedCmpKvHostData(B, static_cast<int32_t>(cmpKvLen));
    std::vector<int32_t> cmpResidualKvHostData(B);
    for (int64_t i = 0; i < B; i++) {
        cmpResidualKvHostData[i] = seqUsedOriKvHostData[i] % static_cast<int32_t>(cmpRatio);
    }
    std::vector<float> sinksHostData(N1, 1.0f);
    std::vector<int32_t> metadataHostData(1024, 0);
    std::vector<uint16_t> attnOutHostData = MakeBf16Data(attnOutSize, 0.0f);
    std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

    std::mt19937 gen(42);
    for (int64_t t = 0; t < T1; t++) {
        for (int64_t n = 0; n < N2; n++) {
            for (int64_t k = 0; k < K; k++) {
                cmpSparseIndicesHostData[t * N2 * K + n * K + k] = static_cast<int32_t>(gen() % cmpKvLen);
            }
        }
    }

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &oriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &cmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qDescaleHostData, descaleShape, &qDescaleDeviceAddr, aclDataType::ACL_FLOAT, &qDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriKvDescaleHostData, descaleShape, &oriKvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &oriKvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvDescaleHostData, descaleShape, &cmpKvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &cmpKvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpSparseIndicesHostData, cmpSparseIndicesShape, &cmpSparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &cmpSparseIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriBlockTableHostData, oriBlockTableShape, &oriBlockTableDeviceAddr, aclDataType::ACL_INT32,
                          &oriBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpBlockTableHostData, cmpBlockTableShape, &cmpBlockTableDeviceAddr, aclDataType::ACL_INT32,
                          &cmpBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32,
                          &seqUsedOriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqUsedCmpKvHostData, seqUsedCmpKvShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32,
                          &seqUsedCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32,
                          &cmpResidualKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";

    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;

    ret = aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize(
        cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv, seqUsedQ, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv, nullptr,
        nullptr, N1, N2, D, quantMode, B, S1, S2, cmpKvLen, 0, K, cmpRatio, oriMaskMode, cmpMaskMode,
        oriWinLeft, oriWinRight, layoutQ, layoutKv, true, true, metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
              return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnQuantSparseFlashMlaGetWorkspaceSize(
        q, oriKv, cmpKv, qDescale, oriKvDescale, cmpKvDescale, nullptr, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqLensQ, nullptr, nullptr, nullptr, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv, nullptr, nullptr, sinks,
        metadata, quantMode, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ,
        layoutKv, 1, false, attnOut, softmaxLse, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(attnOutShape, &attnOutDeviceAddr);

    aclDestroyTensor(q);
    aclDestroyTensor(oriKv);
    aclDestroyTensor(cmpKv);
    aclDestroyTensor(qDescale);
    aclDestroyTensor(oriKvDescale);
    aclDestroyTensor(cmpKvDescale);
    aclDestroyTensor(cmpSparseIndices);
    aclDestroyTensor(oriBlockTable);
    aclDestroyTensor(cmpBlockTable);
    aclDestroyTensor(cuSeqLensQ);
    aclDestroyTensor(cuSeqLensOriKv);
    aclDestroyTensor(cuSeqLensCmpKv);
    aclDestroyTensor(seqUsedQ);
    aclDestroyTensor(seqUsedOriKv);
    aclDestroyTensor(seqUsedCmpKv);
    aclDestroyTensor(cmpResidualKv);
    aclDestroyTensor(sinks);
    aclDestroyTensor(metadata);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(softmaxLse);

    aclrtFree(qDeviceAddr);
    aclrtFree(oriKvDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    aclrtFree(qDescaleDeviceAddr);
    aclrtFree(oriKvDescaleDeviceAddr);
    aclrtFree(cmpKvDescaleDeviceAddr);
    aclrtFree(cmpSparseIndicesDeviceAddr);
    aclrtFree(oriBlockTableDeviceAddr);
    aclrtFree(cmpBlockTableDeviceAddr);
    if (cuSeqLensQDeviceAddr != nullptr) {
        aclrtFree(cuSeqLensQDeviceAddr);
    }
    if (seqUsedOriKvDeviceAddr != nullptr) {
        aclrtFree(seqUsedOriKvDeviceAddr);
    }
    if (seqUsedCmpKvDeviceAddr != nullptr) {
        aclrtFree(seqUsedCmpKvDeviceAddr);
    }
    if (cmpResidualKvDeviceAddr != nullptr) {
        aclrtFree(cmpResidualKvDeviceAddr);
    }
    aclrtFree(sinksDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
