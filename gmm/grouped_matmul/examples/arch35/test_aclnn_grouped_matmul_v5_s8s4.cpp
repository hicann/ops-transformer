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
 * \file test_aclnn_grouped_matmul_v5_s8s4.cpp
 * \brief Ascend 950 S8S4 per-channel asymmetric quantization example.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"

namespace {

#define CHECK_ACL(expr)                                                                            \
    do {                                                                                           \
        const aclError ret = (expr);                                                               \
        if (ret != ACL_SUCCESS) {                                                                  \
            std::cerr << #expr << " failed, ret=" << ret << ", message=" << aclGetRecentErrMsg() \
                      << std::endl;                                                                \
            return ret;                                                                            \
        }                                                                                          \
    } while (0)

struct TensorResource {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;

    ~TensorResource()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

int CreateTensor(const void *hostData, size_t byteSize, const std::vector<int64_t> &shape, aclDataType dtype,
                 TensorResource &resource)
{
    CHECK_ACL(aclrtMalloc(&resource.deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(resource.deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    const auto strides = MakeStrides(shape);
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), resource.deviceAddr);
    if (resource.tensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    mantissa += 0x1000U;
    if ((mantissa & 0x800000U) != 0) {
        mantissa = 0;
        if (++exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x3ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffU;
            bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int RunS8S4Example(aclrtStream stream)
{
    constexpr int64_t M = 8;
    constexpr int64_t K = 512;
    constexpr int64_t N = 16;
    constexpr int64_t E = 2;

    std::vector<int8_t> x(M * K);
    std::vector<int8_t> logicalWeight(E * K * N);
    for (int64_t i = 0; i < M * K; ++i) {
        x[i] = static_cast<int8_t>((i * 5 + 3) % 16 - 8);
    }
    for (int64_t i = 0; i < E * K * N; ++i) {
        logicalWeight[i] = static_cast<int8_t>((i * 7 + 1) % 16 - 8);
    }

    // ACL_INT4 uses two signed INT4 values per byte. The low nibble stores the even N index.
    std::vector<uint8_t> packedWeight(logicalWeight.size() / 2, 0);
    for (size_t i = 0; i < logicalWeight.size(); ++i) {
        const uint8_t nibble = static_cast<uint8_t>(logicalWeight[i]) & 0x0fU;
        if ((i & 1U) == 0) {
            packedWeight[i / 2] = nibble;
        } else {
            packedWeight[i / 2] |= static_cast<uint8_t>(nibble << 4);
        }
    }

    std::vector<float> floatScale(E * N);
    std::vector<float> fusedOffset(E * N);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t n = 0; n < N; ++n) {
            floatScale[e * N + n] = static_cast<float>(1 + (e + n) % 4) / 128.0F;
            fusedOffset[e * N + n] = static_cast<float>((e + n) % 5 - 2) * floatScale[e * N + n];
        }
    }
    // UINT64 FixPipe scale encoding used by aclnnTransQuantParam when offsetArray is null:
    // bits[31:0] hold the FLOAT32 scale and bit 46 selects the M1 quantization mode.
    std::vector<uint64_t> quantScale(floatScale.size());
    for (size_t i = 0; i < floatScale.size(); ++i) {
        uint32_t scaleBits = 0;
        std::memcpy(&scaleBits, &floatScale[i], sizeof(scaleBits));
        quantScale[i] = static_cast<uint64_t>(scaleBits) | (1ULL << 46);
    }

    // S8S4 API bias is the offline INT4 conversion correction:
    // reduce_sum(8 * weight * scale, K). It is not an ordinary matmul bias.
    std::vector<float> bias(E * N, 0.0F);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                bias[e * N + n] += 8.0F * logicalWeight[(e * K + k) * N + n] * floatScale[e * N + n];
            }
        }
    }
    std::vector<float> perTokenScale(M, 1.0F);
    std::vector<int64_t> groupList = {M / 2, M / 2}; // groupListType=1: token count of each expert.
    std::vector<uint16_t> output(M * N, FloatToHalf(0.0F));

    TensorResource xTensor;
    TensorResource weightTensor;
    TensorResource biasTensor;
    TensorResource scaleTensor;
    TensorResource offsetTensor;
    TensorResource perTokenTensor;
    TensorResource groupListTensor;
    TensorResource outputTensor;
    CHECK_ACL(CreateTensor(x.data(), x.size(), {M, K}, ACL_INT8, xTensor));
    CHECK_ACL(CreateTensor(packedWeight.data(), packedWeight.size(), {E, K, N}, ACL_INT4, weightTensor));
    CHECK_ACL(CreateTensor(bias.data(), bias.size() * sizeof(float), {E, N}, ACL_FLOAT, biasTensor));
    CHECK_ACL(CreateTensor(quantScale.data(), quantScale.size() * sizeof(uint64_t), {E, 1, N}, ACL_UINT64,
                           scaleTensor));
    CHECK_ACL(CreateTensor(fusedOffset.data(), fusedOffset.size() * sizeof(float), {E, 1, N}, ACL_FLOAT,
                           offsetTensor));
    CHECK_ACL(CreateTensor(perTokenScale.data(), perTokenScale.size() * sizeof(float), {M}, ACL_FLOAT,
                           perTokenTensor));
    CHECK_ACL(CreateTensor(groupList.data(), groupList.size() * sizeof(int64_t), {E}, ACL_INT64, groupListTensor));
    CHECK_ACL(CreateTensor(output.data(), output.size() * sizeof(uint16_t), {M, N}, ACL_FLOAT16, outputTensor));

    const aclTensor *xItems[] = {xTensor.tensor};
    const aclTensor *weightItems[] = {weightTensor.tensor};
    const aclTensor *biasItems[] = {biasTensor.tensor};
    const aclTensor *scaleItems[] = {scaleTensor.tensor};
    const aclTensor *offsetItems[] = {offsetTensor.tensor};
    const aclTensor *perTokenItems[] = {perTokenTensor.tensor};
    const aclTensor *outputItems[] = {outputTensor.tensor};
    aclTensorList *xList = aclCreateTensorList(xItems, 1);
    aclTensorList *weightList = aclCreateTensorList(weightItems, 1);
    aclTensorList *biasList = aclCreateTensorList(biasItems, 1);
    aclTensorList *scaleList = aclCreateTensorList(scaleItems, 1);
    aclTensorList *offsetList = aclCreateTensorList(offsetItems, 1);
    aclTensorList *perTokenList = aclCreateTensorList(perTokenItems, 1);
    aclTensorList *outputList = aclCreateTensorList(outputItems, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulV5GetWorkspaceSize(
        xList, weightList, biasList, scaleList, offsetList, nullptr, nullptr, perTokenList, groupListTensor.tensor,
        nullptr, nullptr, nullptr, 3, 0, 1, 0, nullptr, outputList, nullptr, nullptr, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclnnGroupedMatmulV5GetWorkspaceSize failed, ret=" << ret
                  << ", message=" << aclGetRecentErrMsg() << std::endl;
        return ret;
    }

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnGroupedMatmulV5(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputTensor.deviceAddr,
                          output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    aclDestroyTensorList(xList);
    aclDestroyTensorList(weightList);
    aclDestroyTensorList(biasList);
    aclDestroyTensorList(scaleList);
    aclDestroyTensorList(offsetList);
    aclDestroyTensorList(perTokenList);
    aclDestroyTensorList(outputList);

    float maxError = 0.0F;
    int64_t rowStart = 0;
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t m = rowStart; m < rowStart + groupList[e]; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float golden = 0.0F;
                for (int64_t k = 0; k < K; ++k) {
                    const float dequantWeight = logicalWeight[(e * K + k) * N + n] * floatScale[e * N + n] +
                                                fusedOffset[e * N + n];
                    golden += static_cast<float>(x[m * K + k]) * dequantWeight;
                }
                golden *= perTokenScale[m];
                const float actual = HalfToFloat(output[m * N + n]);
                maxError = std::max(maxError, std::fabs(actual - golden));
            }
        }
        rowStart += groupList[e];
    }
    std::cout << "S8S4 per-channel offset example: max_abs_error=" << maxError << std::endl;
    return maxError < 0.2F ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    const int ret = RunS8S4Example(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::cout << (ret == ACL_SUCCESS ? "PASS" : "FAIL") << std::endl;
    return ret == ACL_SUCCESS ? 0 : 1;
}
