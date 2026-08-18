/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
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
    return (a + b - 1) / b;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ret=%d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ret=%d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ret=%d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }

    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &logicalShape,
                    const std::vector<int64_t> &storageShape, void **deviceAddr, aclDataType dataType,
                    aclFormat formatType, aclTensor **tensor)
{
    uint64_t size = GetShapeSize(storageShape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ret=%d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ret=%d\n", ret); return ret);
    std::vector<int64_t> strides(logicalShape.size(), 1);
    for (int64_t i = logicalShape.size() - 2; i >= 0; --i) {
        strides[i] = logicalShape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0, formatType,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorND(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensor(hostData, shape, shape, deviceAddr, dataType, ACL_FORMAT_ND, tensor);
}

template <typename T>
void PrintVector(const std::vector<T> &data, const std::string &name, int64_t printNum = 32)
{
    LOG_PRINT("======== %s ========\n", name.c_str());
    int64_t size = static_cast<int64_t>(data.size());
    int64_t limit = std::min(size, printNum);
    for (int64_t i = 0; i < limit; ++i) {
        LOG_PRINT("%s[%ld] = %d\n", name.c_str(), i, static_cast<int32_t>(data[i]));
    }
    LOG_PRINT("\n");
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return ret);
    int64_t E = 4;
    int64_t M = 512;
    int64_t N = 7168;
    int64_t K = 2048;

    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightLogicalShape = {E, K, N};
    std::vector<int64_t> weightStorageShape = {E, K / 16, N / 32, 16, 32};
    std::vector<int64_t> weightScaleShape = {E, Ceil(K, 64), N, 2};
    std::vector<int64_t> xScaleShape = {M, Ceil(K, 64), 2};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M, Ceil(N / 2, 64), 2};

    std::vector<uint8_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShape), 1);
    std::vector<int8_t> weightScaleHostData(GetShapeSize(weightScaleShape), 1);
    std::vector<int8_t> xScaleHostData(GetShapeSize(xScaleShape), 1);
    // std::vector<int64_t> groupListHostData(E, 1);
    std::vector<int64_t> groupListHostData = {93, 188, 76, 155};
    std::vector<uint8_t> outputHostData(GetShapeSize(outputShape), 0);
    std::vector<int8_t> outputScaleHostData(GetShapeSize(outputScaleShape), 0);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *outputScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weightTensor = nullptr;
    aclTensor *weightScaleTensor = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *output = nullptr;
    aclTensor *outputScale = nullptr;

    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;

    ret = CreateAclTensorND<uint8_t>(xHostData, xShape, &xDeviceAddr, ACL_FLOAT8_E4M3FN, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(weightHostData, weightLogicalShape, weightStorageShape, &weightDeviceAddr,
                                   ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, &weightTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weightTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor *weightArray[] = {weightTensor};
    weight = aclCreateTensorList(weightArray, 1);
    CHECK_RET(weight != nullptr, LOG_PRINT("aclCreateTensorList(weight) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                               aclDestroyTensorList);
    ret = CreateAclTensorND<int8_t>(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, ACL_FLOAT8_E8M0,
                                    &weightScaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightScaleTensorPtr(weightScaleTensor,
                                                                                        aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor *weightScaleArray[] = {weightScaleTensor};
    weightScale = aclCreateTensorList(weightScaleArray, 1);
    CHECK_RET(weightScale != nullptr, LOG_PRINT("aclCreateTensorList(weightScale) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);

    ret = CreateAclTensorND<int8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, ACL_FLOAT8_E8M0, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, ACL_FLOAT8_E4M3FN, &output);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<int8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, ACL_FLOAT8_E8M0,
                                    &outputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(x, weight, weightScale, nullptr, nullptr, xScale,
                                                                  nullptr, groupList, 2, 0, 2, 1, nullptr, output,
                                                                  outputScale, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ret=%d\n", ret); return ret);
    LOG_PRINT("workspaceSize = %lu\n", workspaceSize);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed\n"); return ret);
    }

    ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("run op failed. ret=%d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("sync stream failed\n"); return ret);
    LOG_PRINT("run success\n");

    ret = aclrtMemcpy(outputHostData.data(), outputHostData.size() * sizeof(uint8_t), outputDeviceAddr,
                      outputHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ret=%d\n", ret); return ret);

    ret = aclrtMemcpy(outputScaleHostData.data(), outputScaleHostData.size() * sizeof(int8_t), outputScaleDeviceAddr,
                      outputScaleHostData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScale failed. ret=%d\n", ret); return ret);

    PrintVector<uint8_t>(outputHostData, "output", 64);

    PrintVector<int8_t>(outputScaleHostData, "outputScale", 64);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // Release ACL objects and device buffers before resetting the device in Finalize.
    weightScaleTensorListPtr.reset();
    weightTensorListPtr.reset();
    outputScaleTensorPtr.reset();
    outputTensorPtr.reset();
    groupListTensorPtr.reset();
    xScaleTensorPtr.reset();
    weightScaleTensorPtr.reset();
    weightTensorPtr.reset();
    xTensorPtr.reset();

    outputScaleDeviceAddrPtr.reset();
    outputDeviceAddrPtr.reset();
    groupListDeviceAddrPtr.reset();
    xScaleDeviceAddrPtr.reset();
    weightScaleDeviceAddrPtr.reset();
    weightDeviceAddrPtr.reset();
    xDeviceAddrPtr.reset();

    Finalize(deviceId, stream);
    return 0;
}
