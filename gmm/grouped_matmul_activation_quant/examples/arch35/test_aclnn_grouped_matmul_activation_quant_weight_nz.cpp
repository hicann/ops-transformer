/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_activation_quant_weight_nz.h"

/*!
 * \file test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp
 * \brief GroupedMatmulActivationQuant WeightNz MXFP8 aclnn example for Ascend 950PR/Ascend 950DT.
 */

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>>& hostData, const std::vector<std::vector<int64_t>>& shapes,
                        void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList** tensorList)
{
    int size = static_cast<int>(shapes.size());
    std::vector<aclTensor*> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, formatType, &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListWithStorageShape(const std::vector<std::vector<T>>& hostData,
                                        const std::vector<std::vector<int64_t>>& viewShapes,
                                        const std::vector<std::vector<int64_t>>& storageShapes,
                                        const std::vector<std::vector<int64_t>>& strides, void** deviceAddr,
                                        aclDataType dataType, aclFormat formatType, aclTensorList** tensorList)
{
    int size = static_cast<int>(storageShapes.size());
    std::vector<aclTensor*> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        auto storageSize = GetShapeSize(storageShapes[i]) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType, strides[i].data(), 0,
                                     formatType, storageShapes[i].data(), storageShapes[i].size(), *(deviceAddr + i));
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int RunGroupedMatmulActivationQuantWeightNz(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t e = 2;
    int64_t m = 4;
    int64_t k = 64;
    int64_t n = 128;
    int64_t kScaleBlock = CeilDiv(k, 64);
    int64_t nScaleBlock = CeilDiv(n, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> groupListShape = {e};
    std::vector<std::vector<int64_t>> weightViewShapes = {{e, k, n}};
    std::vector<std::vector<int64_t>> weightStorageShapes = {{e, CeilDiv(n, 32), CeilDiv(k, 16), 16, 32}};
    std::vector<std::vector<int64_t>> weightStrides = {{k * n, n, 1}};
    std::vector<std::vector<int64_t>> weightScaleShapes = {{e, kScaleBlock, n, 2}};
    std::vector<int64_t> xScaleShape = {m, kScaleBlock, 2};
    std::vector<int64_t> yShape = {m, n};
    std::vector<int64_t> yScaleShape = {m, nScaleBlock, 2};

    std::vector<uint8_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int64_t> groupListHostData = {2, 2};
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShapes[0]), 1);
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShapes[0]), 1);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 1);
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    void* xDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupList = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* weightScale = nullptr;
    aclTensorList* bias = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                   aclFormat::ACL_FORMAT_ND, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   aclFormat::ACL_FORMAT_ND, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
    ret = CreateAclTensorListWithStorageShape<uint8_t>(weightHostDataList, weightViewShapes, weightStorageShapes,
                                                       weightStrides, &weightDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                                       aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightTensorListPtr(weight,
                                                                                              aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
    ret = CreateAclTensorList<uint8_t>(weightScaleHostDataList, weightScaleShapes, &weightScaleDeviceAddr,
                                       aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &weightScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                   aclFormat::ACL_FORMAT_ND, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                   aclFormat::ACL_FORMAT_ND, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                   aclFormat::ACL_FORMAT_ND, &yScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yScaleTensorPtr(yScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const char* activationType = "gelu_tanh";
    int64_t groupListType = 1;
    aclIntArray* tuningConfig = nullptr;
    const char* quantMode = "mx";
    const char* roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 0.0F;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    ret = aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
        x, groupList, weight, weightScale, bias, xScale, activationType, groupListType, tuningConfig, quantMode,
        roundMode, scaleAlg, dstTypeMax, y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnGroupedMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("y shape is: [%ld, %ld]\n", yShape[0], yShape[1]);
    LOG_PRINT("yScale shape is: [%ld, %ld, %ld]\n", yScaleShape[0], yScaleShape[1], yScaleShape[2]);
    LOG_PRINT("GroupedMatmulActivationQuantWeightNz example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunGroupedMatmulActivationQuantWeightNz(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
