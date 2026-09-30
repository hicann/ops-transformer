/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

namespace {
constexpr int64_t SHAPE_PRODUCT_IDENTITY = 1;
constexpr int64_t ELEMENT_STRIDE = 1;
constexpr int64_t LAST_DIM_OFFSET = 1;
constexpr int64_t PENULTIMATE_DIM_OFFSET = 2;
constexpr int64_t DEFAULT_PRINT_ELEMENTS = 32;
constexpr int64_t OUTPUT_PRINT_ELEMENTS = 64;
constexpr int64_t NZ_K0 = 16;
constexpr int64_t NZ_C0 = 32;
constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t SCALE_PAIR_SIZE = 2;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr size_t SINGLE_TENSOR_COUNT = 1;
constexpr uint8_t E8M0_ONE = 127;
constexpr size_t X_PATTERN_MULTIPLIER = 7;
constexpr size_t X_PATTERN_OFFSET = 3;
constexpr size_t WEIGHT_PATTERN_MULTIPLIER = 5;
constexpr size_t WEIGHT_PATTERN_OFFSET = 1;
constexpr int ARG_TRANSPOSE = 1;
constexpr int ARG_DUMP_PREFIX = 2;
constexpr int ARG_SCALE_ALG = 3;
constexpr int MAX_ARGUMENT_COUNT = 4;
constexpr int INVALID_ARGUMENT_EXIT = 2;
constexpr int64_t SCALE_ALG_CUBLAS = 1;
} // namespace

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = SHAPE_PRODUCT_IDENTITY;
    for (auto v : shape) {
        shapeSize *= v;
    }
    return shapeSize;
}

template <typename T1, typename T2>
auto Ceil(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - LAST_DIM_OFFSET) / b;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ret=%d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ret=%d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ret=%d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& logicalShape,
                    const std::vector<int64_t>& storageShape, void** deviceAddr, aclDataType dataType,
                    aclFormat formatType, aclTensor** tensor, const std::vector<int64_t>* customStrides = nullptr)
{
    uint64_t size = GetShapeSize(storageShape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ret=%d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ret=%d\n", ret); return ret);
    std::vector<int64_t> strides(logicalShape.size(), ELEMENT_STRIDE);
    for (int64_t i = logicalShape.size() - PENULTIMATE_DIM_OFFSET; i >= 0; --i) {
        strides[i] = logicalShape[i + LAST_DIM_OFFSET] * strides[i + LAST_DIM_OFFSET];
    }
    if (customStrides != nullptr) {
        strides = *customStrides;
    }
    *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0, formatType,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
{
    return CreateAclTensor(hostData, shape, shape, deviceAddr, dataType, ACL_FORMAT_ND, tensor);
}

template <typename T>
void PrintVector(const std::vector<T>& data, const std::string& name, int64_t printNum = DEFAULT_PRINT_ELEMENTS)
{
    LOG_PRINT("======== %s ========\n", name.c_str());
    int64_t size = static_cast<int64_t>(data.size());
    int64_t limit = std::min(size, printNum);
    for (int64_t i = 0; i < limit; ++i) {
        LOG_PRINT("%s[%ld] = %d\n", name.c_str(), i, static_cast<int32_t>(data[i]));
    }
    LOG_PRINT("\n");
}

bool DumpFile(const std::string& path, const std::vector<uint8_t>& data)
{
    std::ofstream outputFile(path, std::ios::binary);
    if (!outputFile.is_open()) {
        LOG_PRINT("failed to open dump file: %s\n", path.c_str());
        return false;
    }
    outputFile.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    outputFile.flush();
    if (!outputFile.good()) {
        LOG_PRINT("failed to write or flush dump file: %s\n", path.c_str());
        return false;
    }
    return true;
}

int RunExample(bool transposeWeight, const std::string& dumpPrefix, int64_t scaleAlg, aclrtStream stream)
{
    // 2. 构造输入和输出 Tensor；权重转置时同步调整 weightScale 的 view。
    constexpr int64_t dequantMode = 2;
    const int64_t dequantDtype = 0;
    constexpr int64_t quantMode = 2;
    const int64_t groupListType = 0;
    constexpr int64_t swigluMode = 2;
    constexpr double clampLimit = 7.0;
    constexpr double gluAlpha = 1.702;
    constexpr double gluBias = 1.0;
    const char* roundMode = "rint";
    const double dstTypeMax = 0.0;
    const aclTensorList* weightAssistMatrix = nullptr;
    const aclTensor* bias = nullptr;
    const aclTensor* smoothScale = nullptr;
    const aclIntArray* tuningConfigOptional = nullptr;
    constexpr int64_t E = 1;
    constexpr int64_t M = 2048;
    constexpr int64_t K = 64;
    constexpr int64_t N = 128;

    std::vector<int64_t> xShape = {M, K};
    // The API entry view is always [E, K, N]. A transposed (ZN) weight is
    // distinguished by its view stride and physical FRACTAL_NZ storage.
    std::vector<int64_t> weightLogicalShape = {E, K, N};
    std::vector<int64_t> weightStorageShape =
        transposeWeight ? std::vector<int64_t>{E, Ceil(K, NZ_C0), Ceil(N, NZ_K0), NZ_K0, NZ_C0} :
                          std::vector<int64_t>{E, Ceil(N, NZ_C0), Ceil(K, NZ_K0), NZ_K0, NZ_C0};
    const int64_t scaleK = Ceil(K, MX_GROUP_SIZE);
    std::vector<int64_t> weightScaleShape = {E, scaleK, N, SCALE_PAIR_SIZE};
    std::vector<int64_t> weightScaleStorageShape =
        transposeWeight ? std::vector<int64_t>{E, N, scaleK, SCALE_PAIR_SIZE} : weightScaleShape;
    std::vector<int64_t> xScaleShape = {M, Ceil(K, MX_GROUP_SIZE), SCALE_PAIR_SIZE};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / SWIGLU_SPLIT_FACTOR};
    std::vector<int64_t> outputScaleShape = {M, Ceil(N / SWIGLU_SPLIT_FACTOR, MX_GROUP_SIZE), SCALE_PAIR_SIZE};

    constexpr uint8_t fp8Pattern[] = {0x38, 0xb8, 0x40, 0x30, 0xc0, 0x28, 0x3c, 0xbc, 0x48, 0xc8, 0x50, 0x20};
    std::vector<uint8_t> xHostData(GetShapeSize(xShape));
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShape));
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] =
            fp8Pattern[(i * X_PATTERN_MULTIPLIER + X_PATTERN_OFFSET) % (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
    }
    for (size_t i = 0; i < weightHostData.size(); ++i) {
        weightHostData[i] = fp8Pattern[(i * WEIGHT_PATTERN_MULTIPLIER + WEIGHT_PATTERN_OFFSET) %
                                       (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
    }
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShape), E8M0_ONE);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), E8M0_ONE);
    std::vector<int64_t> groupListHostData = {M};
    std::vector<uint8_t> outputHostData(GetShapeSize(outputShape), 0);
    std::vector<uint8_t> outputScaleHostData(GetShapeSize(outputScaleShape), 0);

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* weightScaleTensor = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;

    aclTensorList* weight = nullptr;
    aclTensorList* weightScale = nullptr;

    auto ret = CreateAclTensorND<uint8_t>(xHostData, xShape, &xDeviceAddr, ACL_FLOAT8_E4M3FN, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> weightViewStrides = transposeWeight ? std::vector<int64_t>{K * N, ELEMENT_STRIDE, K} :
                                                               std::vector<int64_t>{K * N, N, ELEMENT_STRIDE};
    ret = CreateAclTensor<uint8_t>(weightHostData, weightLogicalShape, weightStorageShape, &weightDeviceAddr,
                                   ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, &weightTensor, &weightViewStrides);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weightTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* weightArray[] = {weightTensor};
    weight = aclCreateTensorList(weightArray, SINGLE_TENSOR_COUNT);
    CHECK_RET(weight != nullptr, LOG_PRINT("aclCreateTensorList(weight) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightTensorListPtr(weight,
                                                                                              aclDestroyTensorList);
    // aclDestroyTensorList also destroys its member tensors.
    weightTensorPtr.release();
    // The ZN scale view is obtained by transposing an [E, N, scaleK, 2]
    // source. Its API entry view remains [E, scaleK, N, 2].
    std::vector<int64_t> weightScaleViewStrides =
        transposeWeight ?
            std::vector<int64_t>{N * scaleK * SCALE_PAIR_SIZE, SCALE_PAIR_SIZE, scaleK * SCALE_PAIR_SIZE,
                                 ELEMENT_STRIDE} :
            std::vector<int64_t>{scaleK * N * SCALE_PAIR_SIZE, N * SCALE_PAIR_SIZE, SCALE_PAIR_SIZE, ELEMENT_STRIDE};
    ret =
        CreateAclTensor<uint8_t>(weightScaleHostData, weightScaleShape, weightScaleStorageShape, &weightScaleDeviceAddr,
                                 ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleTensor, &weightScaleViewStrides);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScaleTensorPtr(weightScaleTensor,
                                                                                       aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* weightScaleArray[] = {weightScaleTensor};
    weightScale = aclCreateTensorList(weightScaleArray, SINGLE_TENSOR_COUNT);
    CHECK_RET(weightScale != nullptr, LOG_PRINT("aclCreateTensorList(weightScale) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    weightScaleTensorPtr.release();

    ret = CreateAclTensorND<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, ACL_FLOAT8_E8M0, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, ACL_FLOAT8_E4M3FN, &output);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputTensorPtr(output, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, ACL_FLOAT8_E8M0,
                                     &outputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 第一段接口完成校验并获取 workspace 大小和 executor。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
        quantMode, groupListType, tuningConfigOptional, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg,
        dstTypeMax, output, outputScale, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ret=%d\n", ret); return ret);
    LOG_PRINT("workspaceSize = %lu\n", workspaceSize);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed\n"); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspacePtr(workspaceAddr, aclrtFree);

    // 4. 第二段接口执行算子，随后同步并读取输出。
    ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspaceAddr, workspaceSize, executor, stream);
    // Synchronize even if launch reports an error, before releasing device buffers.
    const auto syncRet = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("run op failed. ret=%d\n", ret); return ret);
    CHECK_RET(syncRet == ACL_SUCCESS, LOG_PRINT("sync stream failed. ret=%d\n", syncRet); return syncRet);

    LOG_PRINT("run success\n");

    ret = aclrtMemcpy(outputHostData.data(), outputHostData.size() * sizeof(uint8_t), outputDeviceAddr,
                      outputHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ret=%d\n", ret); return ret);

    ret = aclrtMemcpy(outputScaleHostData.data(), outputScaleHostData.size() * sizeof(uint8_t), outputScaleDeviceAddr,
                      outputScaleHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScale failed. ret=%d\n", ret); return ret);

    PrintVector<uint8_t>(outputHostData, "output", MX_GROUP_SIZE);

    PrintVector<uint8_t>(outputScaleHostData, "outputScale", MX_GROUP_SIZE);

    if (!dumpPrefix.empty()) {
        CHECK_RET(DumpFile(dumpPrefix + "_output.bin", outputHostData), return ACL_ERROR_FAILURE);
        CHECK_RET(DumpFile(dumpPrefix + "_output_scale.bin", outputScaleHostData), return ACL_ERROR_FAILURE);
        LOG_PRINT("dumpPrefix=%s\n", dumpPrefix.c_str());
    }

    // Tensor lists, tensors, workspace and device buffers are released by RAII
    // before main destroys the stream and resets the device.
    return ACL_SUCCESS;
}

int main(int argc, char** argv)
{
    bool transposeWeight = false;
    int64_t scaleAlg = 0;
    const std::string dumpPrefix = argc > ARG_DUMP_PREFIX ? argv[ARG_DUMP_PREFIX] : "";
    try {
        const std::string transposeArg = argc > ARG_TRANSPOSE ? argv[ARG_TRANSPOSE] : "0";
        const std::string scaleAlgArg = argc > ARG_SCALE_ALG ? argv[ARG_SCALE_ALG] : "0";
        if (argc > MAX_ARGUMENT_COUNT || (transposeArg != "0" && transposeArg != "1") ||
            (scaleAlgArg != "0" && scaleAlgArg != "1")) {
            LOG_PRINT("usage: %s [transposeWeight: 0|1] [dumpPrefix] [scaleAlg: 0|1]\n", argv[0]);
            return INVALID_ARGUMENT_EXIT;
        }
        transposeWeight = transposeArg == "1";
        scaleAlg = scaleAlgArg == "1" ? SCALE_ALG_CUBLAS : 0;
    } catch (const std::exception& error) {
        LOG_PRINT("invalid arguments: %s\n", error.what());
        return INVALID_ARGUMENT_EXIT;
    }

    // 1. 初始化 Device 和 Stream。
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    try {
        ret = RunExample(transposeWeight, dumpPrefix, scaleAlg, stream);
    } catch (const std::exception& error) {
        LOG_PRINT("example failed: %s\n", error.what());
        ret = ACL_ERROR_FAILURE;
    }

    // 5. 释放 Device 和 Stream；Tensor、TensorList 与 workspace 在 RunExample 中释放。
    const auto destroyRet = aclrtDestroyStream(stream);
    const auto resetRet = aclrtResetDevice(deviceId);
    const auto finalizeRet = aclFinalize();
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(destroyRet == ACL_SUCCESS, LOG_PRINT("aclrtDestroyStream failed. ret=%d\n", destroyRet);
              return destroyRet);
    CHECK_RET(resetRet == ACL_SUCCESS, LOG_PRINT("aclrtResetDevice failed. ret=%d\n", resetRet); return resetRet);
    CHECK_RET(finalizeRet == ACL_SUCCESS, LOG_PRINT("aclFinalize failed. ret=%d\n", finalizeRet); return finalizeRet);
    LOG_PRINT("resource cleanup success\n");
    return ACL_SUCCESS;
}
