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
 * \file test_aclnn_grouped_matmul_weight_nz_s8s4_special.cpp
 * \brief Ascend 950 S8S4 per-channel quantization with a FRACTAL_NZ special ENK
 *        weight, driven through aclnnGroupedMatmulWeightNz.
 *
 * Interface note:
 *   This example uses the NZ entry point aclnnGroupedMatmulWeightNz. tuningConfigOptional is
 *   {expectedTokenNum, specialWeightFormat = 1}; tuningConfig[1] == 1 selects the S8S4 special
 *   ENK-NZ template, which requires FRACTAL_NZ weight, null offset and
 *   groupListType == 1 (see
 *   gmm/grouped_matmul/op_api/grouped_matmul_weight_quant_950_checker.cpp).
 *
 * Weight preparation:
 *   ND INT32 values in ENK order -> FormatCast to NZ -> ConvertWeightToINT4Pack.
 *   The final EKN view shares the packed NZ buffer; no data transpose is performed.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <functional>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_convert_weight_to_int4_pack.h"

namespace {

#define CHECK_ACL(expr) \
    do { \
        const aclError ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            std::cerr << #expr << " failed, ret=" << ret << ", message=" << aclGetRecentErrMsg() << std::endl; \
            return ret; \
        } \
    } while (0)

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    aclTensorList* tensorList = nullptr;

    aclTensorList* CreateList()
    {
        const aclTensor* items[] = {tensor};
        tensorList = aclCreateTensorList(items, 1);
        return tensorList;
    }

    ~TensorResource()
    {
        // Destroying a TensorList also destroys its member tensors.
        if (tensorList != nullptr) {
            aclDestroyTensorList(tensorList);
        } else if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

int CreateTensor(const void* hostData, size_t byteSize, const std::vector<int64_t>& shape, aclDataType dtype,
                 TensorResource& resource)
{
    CHECK_ACL(aclrtMalloc(&resource.deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(resource.deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    const auto strides = MakeStrides(shape);
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.deviceAddr);
    if (resource.tensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

// FormatCast determines the NZ layout. Pack its physical values without another layout conversion,
// then expose the same allocation as an EKN tensor for the special GMM template.
int CreateSpecialWeightNzInt4(const std::vector<int8_t>& hostData, int64_t E, int64_t N, int64_t K,
                              TensorResource& resource, bool packedInt32, aclrtStream stream)
{
    const std::vector<int32_t> weight(hostData.begin(), hostData.end());
    const std::vector<int64_t> enkShape = {E, N, K};
    TensorResource nd;
    CHECK_ACL(CreateTensor(weight.data(), weight.size() * sizeof(int32_t), enkShape, ACL_INT32, nd));

    int64_t* inferredShape = nullptr;
    uint64_t shapeSize = 0;
    int actualFormat = 0;
    CHECK_ACL(aclnnNpuFormatCastCalculateSizeAndFormat(nd.tensor, ACL_FORMAT_FRACTAL_NZ, ACL_INT8, &inferredShape,
                                                       &shapeSize, &actualFormat));
    std::vector<int64_t> nzShape(inferredShape, inferredShape + shapeSize);
    delete[] inferredShape;
    const int64_t elements = std::accumulate(nzShape.begin(), nzShape.end(), int64_t{1}, std::multiplies<int64_t>());
    auto makeView = [](void* address, const std::vector<int64_t>& view, aclDataType dtype, aclFormat format,
                       const std::vector<int64_t>& storage) {
        const auto strides = MakeStrides(view);
        return aclCreateTensor(view.data(), view.size(), dtype, strides.data(), 0, format, storage.data(),
                               storage.size(), address);
    };
    TensorResource nz;
    CHECK_ACL(aclrtMalloc(&nz.deviceAddr, elements * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    nz.tensor = makeView(nz.deviceAddr, enkShape, ACL_INT32, static_cast<aclFormat>(actualFormat), nzShape);
    if (nz.tensor == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnNpuFormatCastGetWorkspaceSize(nd.tensor, nz.tensor, &workspaceSize, &executor));
    TensorResource castWorkspace;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&castWorkspace.deviceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnNpuFormatCast(castWorkspace.deviceAddr, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ND aliases describe the physical NZ sequence so that INT4Pack only compresses values.
    // These descriptors borrow the allocations owned by nz and resource.
    const int64_t packing = packedInt32 ? 8 : 1;
    const std::vector<int64_t> flatShape = {elements / nzShape.back(), nzShape.back()};
    const std::vector<int64_t> packedShape = {flatShape[0], flatShape[1] / packing};
    const aclDataType packedDtype = packedInt32 ? ACL_INT32 : ACL_INT4;
    TensorResource flat;
    TensorResource packed;
    flat.tensor = makeView(nz.deviceAddr, flatShape, ACL_INT32, ACL_FORMAT_ND, flatShape);
    CHECK_ACL(aclrtMalloc(&resource.deviceAddr, elements / 2, ACL_MEM_MALLOC_HUGE_FIRST));
    packed.tensor = makeView(resource.deviceAddr, packedShape, packedDtype, ACL_FORMAT_ND, packedShape);
    if (flat.tensor == nullptr || packed.tensor == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    workspaceSize = 0;
    executor = nullptr;
    CHECK_ACL(aclnnConvertWeightToINT4PackGetWorkspaceSize(flat.tensor, packed.tensor, &workspaceSize, &executor));
    TensorResource packWorkspace;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&packWorkspace.deviceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnConvertWeightToINT4Pack(packWorkspace.deviceAddr, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    nzShape.back() /= packing;
    const std::vector<int64_t> viewShape = {E, K, N / packing};
    resource.tensor = makeView(resource.deviceAddr, viewShape, packedDtype, ACL_FORMAT_FRACTAL_NZ, nzShape);
    if (resource.tensor == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    std::cout << "FormatCast + INT4Pack: ENK -> EKN view, shared buffer=" << resource.deviceAddr << std::endl;
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

int RunS8S4SpecialNzExample(aclrtStream stream, bool packedInt32)
{
    constexpr int64_t M = 16;
    constexpr int64_t K = 512;
    constexpr int64_t N = 256;
    constexpr int64_t E = 2;

    // x is INT8 [M, K].
    std::vector<int8_t> x(M * K);
    for (int64_t i = 0; i < M * K; ++i) {
        x[i] = static_cast<int8_t>((i * 5 + i / 17 + 3) % 16 - 8);
    }

    // Host reference data is ENK; the tensor view is EKN. Values are signed INT4 in [-8, 7].
    std::vector<int8_t> logicalWeight(E * N * K);
    for (int64_t i = 0; i < E * N * K; ++i) {
        logicalWeight[i] = static_cast<int8_t>((i * 7 + i / 19 + i / 1031 + 1) % 16 - 8);
    }

    // Special per-channel quantization uses offset=nullptr.
    std::vector<float> floatScale(E * N);
    std::vector<float> fusedOffset(E * N);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t n = 0; n < N; ++n) {
            floatScale[e * N + n] = static_cast<float>(1 + (e + n) % 4) / 256.0F;
            fusedOffset[e * N + n] = 0.0F;
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
                bias[e * N + n] += 8.0F * logicalWeight[(e * N + n) * K + k] * floatScale[e * N + n];
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
    CHECK_ACL(CreateSpecialWeightNzInt4(logicalWeight, E, N, K, weightTensor, packedInt32, stream));
    CHECK_ACL(CreateTensor(bias.data(), bias.size() * sizeof(float), {E, N}, ACL_FLOAT, biasTensor));
    CHECK_ACL(
        CreateTensor(quantScale.data(), quantScale.size() * sizeof(uint64_t), {E, 1, N}, ACL_UINT64, scaleTensor));
    CHECK_ACL(CreateTensor(fusedOffset.data(), fusedOffset.size() * sizeof(float), {E, 1, N}, ACL_FLOAT, offsetTensor));
    CHECK_ACL(CreateTensor(perTokenScale.data(), perTokenScale.size() * sizeof(float), {M}, ACL_FLOAT, perTokenTensor));
    CHECK_ACL(CreateTensor(groupList.data(), groupList.size() * sizeof(int64_t), {E}, ACL_INT64, groupListTensor));
    CHECK_ACL(CreateTensor(output.data(), output.size() * sizeof(uint16_t), {M, N}, ACL_FLOAT16, outputTensor));

    aclTensorList* xList = xTensor.CreateList();
    aclTensorList* weightList = weightTensor.CreateList();
    aclTensorList* biasList = biasTensor.CreateList();
    aclTensorList* scaleList = scaleTensor.CreateList();
    aclTensorList* offsetList = offsetTensor.CreateList();
    aclTensorList* perTokenList = perTokenTensor.CreateList();
    aclTensorList* outputList = outputTensor.CreateList();
    if (xList == nullptr || weightList == nullptr || biasList == nullptr || scaleList == nullptr ||
        offsetList == nullptr || perTokenList == nullptr || outputList == nullptr) {
        std::cerr << "aclCreateTensorList failed" << std::endl;
        return ACL_ERROR_FAILURE;
    }

    // tuningConfig[1] = 1 selects the S8S4 special ENK-NZ weight format.
    // tuningConfig[0] is the expected token number hint for tiling (0 lets the tiling decide).
    std::vector<int64_t> tuningConfigData = {0, 1};
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigData.data(), tuningConfigData.size());
    if (tuningConfig == nullptr) {
        std::cerr << "aclCreateIntArray failed" << std::endl;
        return ACL_ERROR_FAILURE;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        xList, weightList, biasList, scaleList, nullptr, nullptr, nullptr, perTokenList, groupListTensor.tensor,
        nullptr, nullptr, nullptr, 3, 0, 1, 0, tuningConfig, 0, outputList, nullptr, nullptr, &workspaceSize,
        &executor);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclnnGroupedMatmulWeightNzGetWorkspaceSize failed, ret=" << ret
                  << ", message=" << aclGetRecentErrMsg() << std::endl;
        aclDestroyIntArray(tuningConfig);
        return ret;
    }

    void* workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ret = aclnnGroupedMatmulWeightNz(workspace, workspaceSize, executor, stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyIntArray(tuningConfig);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclnnGroupedMatmulWeightNz failed, ret=" << ret << ", message=" << aclGetRecentErrMsg()
                  << std::endl;
        return ret;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputTensor.deviceAddr,
                          output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    float maxError = 0.0F;
    float maxAbsGolden = 0.0F;
    int64_t rowStart = 0;
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t m = rowStart; m < rowStart + groupList[e]; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float golden = 0.0F;
                for (int64_t k = 0; k < K; ++k) {
                    const float dequantWeight =
                        logicalWeight[(e * N + n) * K + k] * floatScale[e * N + n] + fusedOffset[e * N + n];
                    golden += static_cast<float>(x[m * K + k]) * dequantWeight;
                }
                golden *= perTokenScale[m];
                const float actual = HalfToFloat(output[m * N + n]);
                maxError = std::max(maxError, std::fabs(actual - golden));
                maxAbsGolden = std::max(maxAbsGolden, std::fabs(golden));
            }
        }
        rowStart += groupList[e];
    }

    const float tolerance = 0.002F + 1e-3F * maxAbsGolden;
    std::cout << "S8S4 special ENK-NZ (tuningConfig={0,1}) example: max_abs_error=" << maxError
              << ", max_abs_golden=" << maxAbsGolden << ", tolerance=" << tolerance << std::endl;
    std::cout << "sample output[0]=" << HalfToFloat(output[0]) << ", output[" << (M * N / 2)
              << "]=" << HalfToFloat(output[M * N / 2]) << std::endl;
    return maxError <= tolerance ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

} // namespace

// Usage: example [device_id] [packed_int32: 0 or 1]
int main(int argc, char** argv)
{
    const int deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    const bool packedInt32 = argc > 2 && std::atoi(argv[2]) == 1;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    const int ret = RunS8S4SpecialNzExample(stream, packedInt32);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::cout << (ret == ACL_SUCCESS ? "PASS" : "FAIL") << std::endl;
    return ret == ACL_SUCCESS ? 0 : 1;
}
