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
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

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

int main()
{
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    std::vector<int64_t> queryShape = {1, 2, 2, 16};               // BNSD
    std::vector<int64_t> keyShape = {1, 2, 2, 16};                 // BNSD
    std::vector<int64_t> valueShape = {1, 2, 2, 16};               // BNSD
    std::vector<int64_t> keyAntiquantScaleShape = {1, 2, 1, 16};   // (1, N, 1, D)
    std::vector<int64_t> valueAntiquantScaleShape = {1, 2, 1, 16}; // (1, N, 1, D)
    std::vector<int64_t> outShape = {1, 2, 2, 16};                 // BNSD
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* keyAntiquantScaleDeviceAddr = nullptr;
    void* valueAntiquantScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* keyAntiquantScaleTensor = nullptr;
    aclTensor* valueAntiquantScaleTensor = nullptr;
    aclTensor* outTensor = nullptr;
    int64_t queryShapeSize = GetShapeSize(queryShape);                             // BNSD
    int64_t keyShapeSize = GetShapeSize(keyShape);                                 // BNSD
    int64_t valueShapeSize = GetShapeSize(valueShape);                             // BNSD
    int64_t keyAntiquantScaleShapeSize = GetShapeSize(keyAntiquantScaleShape);     // B, KV_N, ceil(KV_S, 256), 1
    int64_t valueAntiquantScaleShapeSize = GetShapeSize(valueAntiquantScaleShape); // B, KV_N, ceil(KV_S, 256), 1
    int64_t outShapeSize = GetShapeSize(outShape);                                 // BNSD
    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);                   // fp16 1.0
    std::vector<uint8_t> keyHostData(keyShapeSize, 0b00111000);                    // fp8_e4m3 1.0
    std::vector<uint8_t> valueHostData(valueShapeSize, 0b00111000);
    std::vector<uint16_t> keyAntiquantScaleHostData(keyAntiquantScaleShapeSize, 0x3C00);
    std::vector<uint16_t> valueAntiquantScaleHostData(valueAntiquantScaleShapeSize, 0x3C00);
    std::vector<uint16_t> outHostData(outShapeSize, 0x3C00);

    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    aclTensor* tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // Create keyAntiquantScale aclTensor.
    ret = CreateAclTensor(keyAntiquantScaleHostData, keyAntiquantScaleShape, &keyAntiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &keyAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create valueAntiquantScale aclTensor.
    ret = CreateAclTensor(valueAntiquantScaleHostData, valueAntiquantScaleShape, &valueAntiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &valueAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    int64_t numHeads = 2; // N
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1 / sqrt(16); // 1/sqrt(d)
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length() + 1];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 0;
    int64_t pseType = 0;
    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first interface.
    ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, keyAntiquantScaleTensor, nullptr,
        valueAntiquantScaleTensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode,
        innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode,
        pseType, outTensor, nullptr, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it
    // according to the specific API interface definition.
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
    // 6. Release resources.
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(outTensor);
    aclDestroyTensor(keyAntiquantScaleTensor);
    aclDestroyTensor(valueAntiquantScaleTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(keyAntiquantScaleDeviceAddr);
    aclrtFree(valueAntiquantScaleDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
