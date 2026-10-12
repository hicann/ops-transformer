/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_infer_attention_score.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        (void)aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

float fp16_to_float(uint16_t h)
{
    uint32_t sign = ((h >> 15) & 1);
    uint32_t exponent = ((h >> 10) & 0x1F);
    uint32_t mantissa = (h & 0x3FF);

    uint32_t f32;
    if (exponent == 0x1F) {
        exponent = 0xFF;
        mantissa = (mantissa ? 0x7FFFFF : 0);
    } else if (exponent == 0) {
        if (mantissa != 0) {
            exponent = 0x71;
            do {
                mantissa <<= 1;
                exponent--;
            } while (!(mantissa & 0x400));
            mantissa &= 0x3FF;
        }
    } else {
        exponent += (127 - 15);
    }

    f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *((float*)&f32);
}

int aclnnFusedInferAttentionScoreV5FullQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> queryShape = {64, 2, 64}; // TND
    std::vector<int64_t> keyShape = {64, 2, 64};   // TND
    std::vector<int64_t> valueShape = {64, 2, 64}; // TND
    std::vector<int64_t> actualSeqlenVector = {4, 8, 16, 32, 64};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray*)> actualSeqLengthsPtr(actualSeqLengths,
                                                                                          aclDestroyIntArray);
    std::vector<int64_t> actualSeqlenKvVector = {4, 8, 16, 32, 64};
    auto actualSeqLengthsKv = aclCreateIntArray(actualSeqlenKvVector.data(), actualSeqlenKvVector.size());
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray*)> actualSeqLengthsKvPtr(actualSeqLengthsKv,
                                                                                            aclDestroyIntArray);
    std::vector<int64_t> quantScale1Shape = {1};                   // 1
    std::vector<int64_t> keyAntiquantScaleShape = {64, 2, 1, 2};   // （KV_T, KV_N, D/64, 2）
    std::vector<int64_t> valueAntiquantScaleShape = {5, 2, 64, 2}; // （KV_T/64, KV_N, D, 2）
    std::vector<int64_t> dequantScaleQueryShape = {64, 2, 1, 2};   // （Q_T, Q_N, D/64, 2）
    std::vector<int64_t> outShape = {64, 2, 64};                   // TND
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* quantScale1DeviceAddr = nullptr;
    void* dequantScaleQueryDeviceAddr = nullptr;
    void* keyAntiquantScaleDeviceAddr = nullptr;
    void* valueAntiquantScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* quantScale1Tensor = nullptr;
    aclTensor* dequantScaleQueryTensor = nullptr;
    aclTensor* keyAntiquantScaleTensor = nullptr;
    aclTensor* valueAntiquantScaleTensor = nullptr;
    aclTensor* outTensor = nullptr;
    int64_t queryShapeSize = GetShapeSize(queryShape);                             // TND
    int64_t keyShapeSize = GetShapeSize(keyShape);                                 // BNSD
    int64_t valueShapeSize = GetShapeSize(valueShape);                             // BNSD
    int64_t quantScale1ShapeSize = GetShapeSize(quantScale1Shape);                 // 1
    int64_t dequantScaleQueryShapeSize = GetShapeSize(dequantScaleQueryShape);     // （Q_T, Q_N, D/64, 2）
    int64_t keyAntiquantScaleShapeSize = GetShapeSize(keyAntiquantScaleShape);     // （KV_T, KV_N, D/64, 2）
    int64_t valueAntiquantScaleShapeSize = GetShapeSize(valueAntiquantScaleShape); // （KV_T/64, KV_N, D, 2）
    int64_t outShapeSize = GetShapeSize(outShape);                                 // BNSD
    std::vector<uint8_t> queryHostData(queryShapeSize, 0b00111000);                // fp8_e4m3 1.0
    std::vector<uint8_t> keyHostData(keyShapeSize, 0b00111000);
    std::vector<uint8_t> valueHostData(valueShapeSize, 0b00111000);
    std::vector<float> quantScale1HostData(quantScale1ShapeSize, 1);
    std::vector<uint8_t> dequantScaleQueryHostData(dequantScaleQueryShapeSize, 0b01111111); // fp8_e8m0 1.0
    std::vector<uint8_t> keyAntiquantScaleHostData(keyAntiquantScaleShapeSize, 0b01111111);
    std::vector<uint8_t> valueAntiquantScaleHostData(valueAntiquantScaleShapeSize, 0b01111111);
    std::vector<uint16_t> outHostData(outShapeSize, 0x3C00); // fp16 1.0

    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> queryDeviceAddrPtr(queryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> queryTensorPtr(queryTensor, aclDestroyTensor);

    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> keyDeviceAddrPtr(keyDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> keyTensorPtr(keyTensor, aclDestroyTensor);

    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> tensorKeyListPtr(tensorKeyList,
                                                                                           aclDestroyTensorList);

    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> valueDeviceAddrPtr(valueDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> valueTensorPtr(valueTensor, aclDestroyTensor);

    aclTensor* tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> tensorValueListPtr(tensorValueList,
                                                                                             aclDestroyTensorList);

    ret = CreateAclTensor(quantScale1HostData, quantScale1Shape, &quantScale1DeviceAddr, aclDataType::ACL_FLOAT,
                          &quantScale1Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> quantScale1DeviceAddrPtr(quantScale1DeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantScale1TensorPtr(quantScale1Tensor,
                                                                                       aclDestroyTensor);

    ret = CreateAclTensor(keyAntiquantScaleHostData, keyAntiquantScaleShape, &keyAntiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &keyAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> keyAntiquantScaleDeviceAddrPtr(keyAntiquantScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> keyAntiquantScaleTensorPtr(keyAntiquantScaleTensor,
                                                                                             aclDestroyTensor);

    ret = CreateAclTensor(valueAntiquantScaleHostData, valueAntiquantScaleShape, &valueAntiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &valueAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> valueAntiquantScaleDeviceAddrPtr(valueAntiquantScaleDeviceAddr,
                                                                                aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> valueAntiquantScaleTensorPtr(
        valueAntiquantScaleTensor, aclDestroyTensor);

    ret = CreateAclTensor(dequantScaleQueryHostData, dequantScaleQueryShape, &dequantScaleQueryDeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &dequantScaleQueryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> dequantScaleQueryDeviceAddrPtr(dequantScaleQueryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dequantScaleQueryTensorPtr(dequantScaleQueryTensor,
                                                                                             aclDestroyTensor);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(outTensor, aclDestroyTensor);

    int64_t numHeads = 2; // N
    int64_t numKeyValueHeads = 2;
    double scaleValue = 1 / sqrt(64); // 1/sqrt(d)
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    string sLayerOut = "TND";
    char layerOut[sLayerOut.length() + 1];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t queryQuantMode = 6;
    int64_t keyAntiquantMode = 6;
    int64_t valueAntiquantMode = 8;
    int64_t pseType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, actualSeqLengths, actualSeqLengthsKv, nullptr,
        quantScale1Tensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        keyAntiquantScaleTensor, nullptr, valueAntiquantScaleTensor, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, dequantScaleQueryTensor, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens,
        nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, outTensor, nullptr, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        float float_value = fp16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, float_value);
    }
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnFusedInferAttentionScoreV5FullQuantTest(deviceId, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5FullQuantTest failed. ERROR: %d\n", ret);
        Finalize(deviceId, stream);
        return ret;
    }
    Finalize(deviceId, stream);
    return 0;
}
