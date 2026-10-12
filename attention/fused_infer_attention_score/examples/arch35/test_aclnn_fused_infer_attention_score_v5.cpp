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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
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
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> queryShape = {1, 2, 2, 16};
    std::vector<int64_t> keyShape = {1, 2, 2, 16};
    std::vector<int64_t> valueShape = {1, 2, 2, 16};
    std::vector<int64_t> pseShape = {1, 2, 2, 2};
    std::vector<int64_t> attenShape = {1, 1, 2, 2};
    std::vector<int64_t> outShape = {1, 2, 2, 16};
    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t pseShapeSize = GetShapeSize(pseShape);
    int64_t attenShapeSize = GetShapeSize(attenShape);
    int64_t outShapeSize = GetShapeSize(outShape);
    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00); // fp16 1.0
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<uint16_t> valueHostData(valueShapeSize, 0x3C00);
    std::vector<uint16_t> pseHostData(pseShapeSize, 0x3C00); // fp16 1.0
    std::vector<uint8_t> attenHostData(attenShapeSize, 0);   // bool 0 = attend (all positions valid)
    std::vector<uint16_t> outHostData(outShapeSize, 0);
    std::vector<uint16_t> resultData(outShapeSize, 0x3C00);

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* pseDeviceAddr = nullptr;
    void* attenDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* pseTensor = nullptr;
    aclTensor* attenTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensorList* tensorKeyList = nullptr;
    aclTensorList* tensorValueList = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[1] = {nullptr};
    aclTensor* tensorsOfValue[1] = {nullptr};
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1 / sqrt(16);
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
    int keyAntiquantMode = 0;
    int valueAntiquantMode = 0;
    int64_t pseType = 0;
    std::vector<int64_t> actualSeqlenVector = {2};

    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor query failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor key failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    tensorsOfKey[0] = keyTensor;
    tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor value failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    tensorsOfValue[0] = valueTensor;
    tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    ret = CreateAclTensor(pseHostData, pseShape, &pseDeviceAddr, aclDataType::ACL_FLOAT16, &pseTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor pse failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor atten failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateAclTensor out failed. ERROR: %d\n", ret);
        goto RELEASE;
    }

    actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, pseTensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue,
        preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode,
        softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, 0, pseType, outTensor, nullptr, &workspaceSize,
        &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            goto RELEASE;
        }
    }
    ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret);
        goto RELEASE;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        goto RELEASE;
    }

    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      outShapeSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        goto RELEASE;
    }
    for (int64_t i = 0; i < outShapeSize; i++) {
        float float_value = fp16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, float_value);
    }

RELEASE:
    if (queryTensor != nullptr) {
        aclDestroyTensor(queryTensor);
    }
    if (keyTensor != nullptr) {
        aclDestroyTensor(keyTensor);
    }
    if (valueTensor != nullptr) {
        aclDestroyTensor(valueTensor);
    }
    if (pseTensor != nullptr) {
        aclDestroyTensor(pseTensor);
    }
    if (attenTensor != nullptr) {
        aclDestroyTensor(attenTensor);
    }
    if (outTensor != nullptr) {
        aclDestroyTensor(outTensor);
    }
    if (tensorKeyList != nullptr) {
        aclDestroyTensorList(tensorKeyList);
    }
    if (tensorValueList != nullptr) {
        aclDestroyTensorList(tensorValueList);
    }
    if (actualSeqLengths != nullptr) {
        aclDestroyIntArray(actualSeqLengths);
    }
    if (queryDeviceAddr != nullptr) {
        aclrtFree(queryDeviceAddr);
    }
    if (keyDeviceAddr != nullptr) {
        aclrtFree(keyDeviceAddr);
    }
    if (valueDeviceAddr != nullptr) {
        aclrtFree(valueDeviceAddr);
    }
    if (pseDeviceAddr != nullptr) {
        aclrtFree(pseDeviceAddr);
    }
    if (attenDeviceAddr != nullptr) {
        aclrtFree(attenDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
