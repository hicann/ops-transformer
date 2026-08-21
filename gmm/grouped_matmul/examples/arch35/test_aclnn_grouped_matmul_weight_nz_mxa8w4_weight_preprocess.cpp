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
 * \file test_aclnn_grouped_matmul_weight_nz_mxa8w4_weight_preprocess.cpp
 * \brief GMM_MX_A8W4场景调用示例：先预处理weight和weightScale，再将输出直接传给GroupedMatmulWeightNz。
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/float4_e2m1.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

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

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId)
        : deviceId_(deviceId)
    {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (aclInited_) {
            aclFinalize();
        }
    }

    int Init(aclrtStream *stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;
        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool deviceSet_ = false;
    bool aclInited_ = false;
};

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &viewShape,
                    const std::vector<int64_t> &viewStrides, const std::vector<int64_t> &storageShape,
                    aclDataType dataType, aclFormat format, void **deviceAddr, aclTensor **tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    const int64_t *strides = viewStrides.empty() ? nullptr : viewStrides.data();
    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides, 0, format, storageShape.data(),
                              storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

int CreateSingleTensorList(aclTensor *tensor, aclTensorList **tensorList)
{
    aclTensor *tensors[] = {tensor};
    *tensorList = aclCreateTensorList(tensors, 1);
    CHECK_RET(*tensorList != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

std::vector<uint8_t> PackFp4(const std::vector<float> &data)
{
    std::vector<uint8_t> packedData((data.size() + 1) / 2, 0);
    for (size_t i = 0; i < data.size(); i += 2) {
        uint8_t low = op::Float4E2M1(data[i]).value;
        uint8_t high = i + 1 < data.size() ? op::Float4E2M1(data[i + 1]).value : 0;
        packedData[i / 2] = static_cast<uint8_t>((high << 4) | low);
    }
    return packedData;
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    AclRuntimeGuard runtime(deviceId);
    auto ret = runtime.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t groupNum = 2;
    const int64_t m = 64;
    const int64_t k = 128;
    const int64_t n = 64;
    const int64_t kGroupSize = 32;
    const int64_t scaleK = CEIL_DIV(k, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> xStrides = {k, 1};
    std::vector<int64_t> weightViewShape = {groupNum, k, n};
    std::vector<int64_t> weightStorageShape = {groupNum, n, k};
    std::vector<int64_t> weightStrides = {k * n, 1, k};
    std::vector<int64_t> xScaleShape = {m, scaleK, 2};
    std::vector<int64_t> xScaleStrides = {scaleK * 2, 2, 1};
    std::vector<int64_t> weightScaleViewShape = {groupNum, scaleK, n, 2};
    std::vector<int64_t> weightScaleStorageShape = {groupNum, n, scaleK, 2};
    std::vector<int64_t> weightScaleStrides = {n * scaleK * 2, 2, scaleK * 2, 1};
    std::vector<int64_t> outWeightStorageShape = {groupNum, CEIL_DIV(k, 32), CEIL_DIV(n, 16), 16, 32};
    std::vector<int64_t> outShape = {m, n};
    std::vector<int64_t> outStrides = {n, 1};
    std::vector<int64_t> groupListShape = {groupNum};
    std::vector<int64_t> groupListStrides = {1};

    std::vector<uint8_t> xHostData(GetShapeSize(xShape), 0b00111000); // FLOAT8_E4M3FN 1.0
    std::vector<float> weightFloatData(GetShapeSize(weightStorageShape), 1.0f);
    std::vector<uint8_t> weightHostData = PackFp4(weightFloatData);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 0b01111111); // FLOAT8_E8M0 1.0
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleStorageShape),
                                             0b10000101); // FLOAT8_E8M0 64.0
    std::vector<uint8_t> outWeightHostData(GetShapeSize(outWeightStorageShape) / 2, 0);
    std::vector<uint8_t> outWeightScaleHostData(GetShapeSize(weightScaleStorageShape), 0);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<int64_t> groupListHostData = {m / groupNum, m};

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *outWeightDeviceAddr = nullptr;
    void *outWeightScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *weightTensor = nullptr;
    aclTensor *xScaleTensor = nullptr;
    aclTensor *weightScaleTensor = nullptr;
    aclTensor *outWeightTensor = nullptr;
    aclTensor *outWeightScaleTensor = nullptr;
    aclTensor *outTensor = nullptr;
    aclTensor *groupListTensor = nullptr;
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *out = nullptr;

    ret =
        CreateAclTensor(xHostData, xShape, xStrides, xShape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &xDeviceAddr, &xTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorGuard(xTensor, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateSingleTensorList(xTensor, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xPtr(x, aclDestroyTensorList);
    xTensorGuard.release();

    ret = CreateAclTensor(weightHostData, weightViewShape, weightStrides, weightStorageShape, ACL_FLOAT4_E2M1,
                          ACL_FORMAT_ND, &weightDeviceAddr, &weightTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weightTensor, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xScaleHostData, xScaleShape, xScaleStrides, xScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                          &xScaleDeviceAddr, &xScaleTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorGuard(xScaleTensor, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateSingleTensorList(xScaleTensor, &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> perTokenScalePtr(perTokenScale,
                                                                                            aclDestroyTensorList);
    xScaleTensorGuard.release();

    ret = CreateAclTensor(weightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                          ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleDeviceAddr, &weightScaleTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightScaleTensorPtr(weightScaleTensor,
                                                                                        aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outWeightHostData, weightViewShape, weightStrides, outWeightStorageShape, ACL_FLOAT4_E2M1,
                          ACL_FORMAT_FRACTAL_NZ_C0_32, &outWeightDeviceAddr, &outWeightTensor);
    std::unique_ptr<void, aclError (*)(void *)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outWeightTensorGuard(outWeightTensor,
                                                                                        aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateSingleTensorList(outWeightTensor, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outWeightPtr(weight, aclDestroyTensorList);
    outWeightTensorGuard.release();

    ret = CreateAclTensor(outWeightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                          ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &outWeightScaleDeviceAddr, &outWeightScaleTensor);
    std::unique_ptr<void, aclError (*)(void *)> outWeightScaleDeviceAddrPtr(outWeightScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outWeightScaleTensorGuard(outWeightScaleTensor,
                                                                                             aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateSingleTensorList(outWeightScaleTensor, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outWeightScalePtr(antiquantScale,
                                                                                             aclDestroyTensorList);
    outWeightScaleTensorGuard.release();

    ret = CreateAclTensor(outHostData, outShape, outStrides, outShape, ACL_BF16, ACL_FORMAT_ND, &outDeviceAddr,
                          &outTensor);
    std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorGuard(outTensor, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateSingleTensorList(outTensor, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outPtr(out, aclDestroyTensorList);
    outTensorGuard.release();

    ret = CreateAclTensor(groupListHostData, groupListShape, groupListStrides, groupListShape, ACL_INT64, ACL_FORMAT_ND,
                          &groupListDeviceAddr, &groupListTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListPtr(groupListTensor, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weightTensor, weightScaleTensor, nullptr, nullptr,
                                                     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, kGroupSize, outWeightTensor,
                                                     outWeightScaleTensor, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *preprocessWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> preprocessWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&preprocessWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate preprocess workspace failed. ERROR: %d\n", ret); return ret);
        preprocessWorkspacePtr.reset(preprocessWorkspaceAddr);
    }
    ret = aclnnWeightQuantPreprocess(preprocessWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocess failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after preprocess failed. ERROR: %d\n", ret);
              return ret);

    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(x, weight, nullptr, nullptr, nullptr, antiquantScale, nullptr,
                                                     perTokenScale, groupListTensor, nullptr, nullptr, nullptr, 3, 0, 0,
                                                     0, nullptr, 0, out, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *groupedMatmulWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> groupedMatmulWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&groupedMatmulWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate grouped matmul workspace failed. ERROR: %d\n", ret);
                  return ret);
        groupedMatmulWorkspacePtr.reset(groupedMatmulWorkspaceAddr);
    }
    ret = aclnnGroupedMatmulWeightNz(groupedMatmulWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after grouped matmul failed. ERROR: %d\n", ret);
              return ret);

    std::vector<uint16_t> resultData(GetShapeSize(outShape), 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (size_t i = 0; i < std::min<size_t>(resultData.size(), 10); ++i) {
        LOG_PRINT("result[%zu] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    auto ret = AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest failed. ERROR: %d\n", ret);
              return ret);
    return 0;
}
