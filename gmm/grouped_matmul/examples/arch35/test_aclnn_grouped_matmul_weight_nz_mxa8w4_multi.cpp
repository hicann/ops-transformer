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
 * \file test_aclnn_grouped_matmul_weight_nz_mxa8w4_multi.cpp
 * \brief aclnnGroupedMatmulWeightNz 单多单 MxA8W4 伪量化场景调用示例：
 *        x 为单 tensor (FLOAT8_E4M3FN)，weight/antiquantScale/bias 为多 tensor，
 *        y 为单 tensor (BFLOAT16)，weight 为 FLOAT4_E2M1 NZ 转置输入。
 */

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"

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

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1L;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostDataList,
                        const std::vector<std::vector<int64_t>> &shapes, std::vector<void *> &deviceAddrList,
                        aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    std::vector<aclTensor *> tensors(size, nullptr);
    deviceAddrList.resize(size, nullptr);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostDataList[i], shapes[i], &deviceAddrList[i], dataType, &tensors[i]);
        if (ret != ACL_SUCCESS) {
            for (int j = 0; j < i; j++) {
                aclDestroyTensor(tensors[j]);
                aclrtFree(deviceAddrList[j]);
                deviceAddrList[j] = nullptr;
            }
            return ret;
        }
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorNzTransposed(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    int64_t k = shape[0];
    int64_t n = shape[1];
    std::vector<int64_t> transStrides = {1, k};
    std::vector<int64_t> storageShape = {k / 32, n / 16, 16, 32};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, transStrides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorListNzTransposed(const std::vector<std::vector<T>> &hostDataList,
                                    const std::vector<std::vector<int64_t>> &shapes,
                                    std::vector<void *> &deviceAddrList, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    std::vector<aclTensor *> tensors(size, nullptr);
    deviceAddrList.resize(size, nullptr);
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensorNzTransposed<T>(hostDataList[i], shapes[i], &deviceAddrList[i], dataType, &tensors[i]);
        if (ret != ACL_SUCCESS) {
            for (int j = 0; j < i; j++) {
                aclDestroyTensor(tensors[j]);
                aclrtFree(deviceAddrList[j]);
                deviceAddrList[j] = nullptr;
            }
            return ret;
        }
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int aclnnGroupedMatmulWeightNzMxA8W4MultiTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 单多单 MxA8W4 场景参数
    int64_t m = 64L;
    int64_t k = 128L;
    int64_t n = 64L;
    int64_t groupSize = 32L;
    int64_t groupNum = 2L;

    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShapes;
    std::vector<std::vector<int64_t>> biasShapes;
    std::vector<std::vector<int64_t>> antiquantScaleShapes;
    for (int64_t i = 0; i < groupNum; i++) {
        weightShapes.push_back({k, n});
        biasShapes.push_back({n});
        antiquantScaleShapes.push_back({k / groupSize / 2, n, 2});
    }
    std::vector<std::vector<int64_t>> perTokenScaleShape = {{m, k / groupSize / 2, 2}};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {groupNum};
    std::vector<int64_t> groupListData;
    for (int64_t i = 0; i < groupNum; i++) {
        groupListData.push_back(m / groupNum * (i + 1));
    }

    std::vector<void *> xDeviceAddrList;
    std::vector<void *> weightDeviceAddrList;
    std::vector<void *> biasDeviceAddrList;
    std::vector<void *> antiquantScaleDeviceAddrList;
    std::vector<void *> perTokenScaleDeviceAddrList;
    std::vector<void *> yDeviceAddrList;
    void *groupListDeviceAddr = nullptr;

    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *bias = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensor *groupedList = nullptr;
    aclTensorList *out = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;

    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;

    // x: FLOAT8_E4M3FN, 每个 element 1 byte
    std::vector<std::vector<int8_t>> xHostData = {std::vector<int8_t>(m * k)};
    for (int64_t j = 0; j < m * k; j++) {
        xHostData[0][j] = static_cast<int8_t>(j % 0x7F);
    }
    ret = CreateAclTensorList(xHostData, xShape, xDeviceAddrList, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xPtr(x, aclDestroyTensorList);

    // weight: FLOAT4_E2M1, 每个 element 4 bit, 2 elements per byte
    std::vector<std::vector<int8_t>> weightHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        std::vector<int8_t> data(k * n / 2);
        for (int64_t j = 0; j < k * n / 2; j++) {
            data[j] = static_cast<int8_t>((j + i * 37) % 256);
        }
        weightHostDataList.push_back(data);
    }
    ret = CreateAclTensorListNzTransposed(weightHostDataList, weightShapes, weightDeviceAddrList,
                                          aclDataType::ACL_FLOAT4_E2M1, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightPtr(weight, aclDestroyTensorList);

    // bias: BFLOAT16
    std::vector<std::vector<uint16_t>> biasHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        std::vector<uint16_t> data(n);
        for (int64_t j = 0; j < n; j++) {
            data[j] = static_cast<uint16_t>((j + i * 50) % 100 + 1);
        }
        biasHostDataList.push_back(data);
    }
    ret = CreateAclTensorList(biasHostDataList, biasShapes, biasDeviceAddrList, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> biasPtr(bias, aclDestroyTensorList);

    // antiquantScale: FLOAT8_E8M0, 每个 element 1 byte
    std::vector<std::vector<int8_t>> antiquantScaleHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        int64_t scaleSize = k / groupSize / 2 * n * 2;
        std::vector<int8_t> data(scaleSize);
        for (int64_t j = 0; j < scaleSize; j++) {
            data[j] = static_cast<int8_t>((j + i * 5) % 16 + 120);
        }
        antiquantScaleHostDataList.push_back(data);
    }
    ret = CreateAclTensorList(antiquantScaleHostDataList, antiquantScaleShapes, antiquantScaleDeviceAddrList,
                              aclDataType::ACL_FLOAT8_E8M0, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> antiquantScalePtr(antiquantScale,
                                                                                             aclDestroyTensorList);

    // perTokenScale: FLOAT8_E8M0, 单 tensor
    std::vector<std::vector<int8_t>> perTokenScaleHostData = {std::vector<int8_t>(m * k / groupSize / 2 * 2)};
    for (int64_t j = 0; j < m * k / groupSize / 2 * 2; j++) {
        perTokenScaleHostData[0][j] = static_cast<int8_t>(j % 16 + 120);
    }
    ret = CreateAclTensorList(perTokenScaleHostData, perTokenScaleShape, perTokenScaleDeviceAddrList,
                              aclDataType::ACL_FLOAT8_E8M0, &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> perTokenScalePtr(perTokenScale,
                                                                                            aclDestroyTensorList);

    // y: BFLOAT16
    std::vector<std::vector<uint16_t>> yHostData = {std::vector<uint16_t>(m * n, 0)};
    ret = CreateAclTensorList(yHostData, yShape, yDeviceAddrList, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outPtr(out, aclDestroyTensorList);

    // groupList: INT64
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListPtr(groupedList, aclDestroyTensor);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);

    // 调用 aclnnGroupedMatmulWeightNz 第一段接口
    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, 0, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用 aclnnGroupedMatmulWeightNz 第二段接口
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出结果
    auto size = GetShapeSize(yShape[0]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddrList[0], size * sizeof(resultData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < std::min<int64_t>(size, 10); j++) {
        LOG_PRINT("result[%ld] is: %g\n", j, Bf16ToFloat(resultData[j]));
    }

    // 释放 device 资源
    for (auto addr : xDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : weightDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : biasDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : antiquantScaleDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : perTokenScaleDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : yDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    if (groupListDeviceAddr)
        aclrtFree(groupListDeviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedMatmulWeightNzMxA8W4MultiTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulWeightNz MxA8W4 multi test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
