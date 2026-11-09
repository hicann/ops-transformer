/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_matmul_v5_mxfp4_splitk.cpp
 * \brief aclnnGroupedMatmulV5 双侧 MXFP4(FLOAT4_E2M1)
 *        双侧 E8M0 + group_type=2
 *        TN 单单单场景调用示例
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1L;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// 按显式 (viewShape, viewStrides, storageShape) 创建 aclTensor。
// 字节数按 hostData 实际大小申请/拷贝：FP4 两个元素打包为 1 字节，host 侧数据按打包后的字节数给出，
template <typename T>
int CreateAclTensorWithLayout(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                              const std::vector<int64_t>& viewStrides, const std::vector<int64_t>& storageShape,
                              void** deviceAddr, aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, viewStrides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateContiguousAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor)
{
    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return CreateAclTensorWithLayout(hostData, shape, strides, shape, deviceAddr, dataType, tensor);
}

void MakeSingleTensorList(aclTensor* tensor, aclTensorList** tensorList)
{
    *tensorList = aclCreateTensorList(&tensor, 1);
}

int aclnnGroupedMatmulMxFp4SplitKTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // MXFP4 + group_type=2（k 轴分组）TN 场景参数
    constexpr int64_t fp4PackRatio = 2L;       // 2 个 FLOAT4_E2M1 元素占 1 个 字节
    constexpr int64_t mxfpDivisor = 64L;       // 每 64 个 k 对应 scale 的一行（末轴 2 x 32k）
    constexpr int64_t printCount = 10L;        // 每组打印结果前 N 个元素
    int64_t m = 128L;                          // 偶数：TN 下 x 的 FP4 打包轴
    int64_t n = 256L;                          // 偶数：weight 的 FP4 打包轴
    std::vector<int64_t> kGroups = {64L, 96L}; // 各组 k 长度，允许任意非负整数
    int64_t groupNum = static_cast<int64_t>(kGroups.size());
    int64_t k = 0L; // 归约总长 k = Σ k_g
    for (auto kG : kGroups) {
        k += kG;
    }
    int64_t S = k / mxfpDivisor + groupNum; // scale 的 K 轴行数 S = floor(k/64) + G

    // view shape
    std::vector<int64_t> xViewShape = {m, k}; // 转置视图：transposeX = true
    std::vector<int64_t> xStrides = {1L, m};  // 物理 [k,m] 连续，m 为打包轴
    std::vector<int64_t> xStorageShape = {k, m};
    std::vector<int64_t> weightShape = {k, n};               // 不转置，ND
    std::vector<int64_t> pertokenViewShape = {m, S, 2L};     // 转置视图
    std::vector<int64_t> pertokenStrides = {2L, m * 2L, 1L}; // 物理 [S,m,2] 连续
    std::vector<int64_t> pertokenStorageShape = {S, m, 2L};
    std::vector<int64_t> scaleShape = {S, n, 2L};   // 不转置
    std::vector<int64_t> yShape = {groupNum, m, n}; // [G,m,n]
    std::vector<int64_t> groupListShape = {groupNum};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* pertokenDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;

    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* perTokenScale = nullptr;
    aclTensorList* activationInput = nullptr;
    aclTensorList* activationQuantScale = nullptr;
    aclTensorList* activationQuantOffset = nullptr;
    aclTensor* groupedList = nullptr;
    aclTensorList* out = nullptr;
    aclTensorList* activationFeatureOut = nullptr;
    aclTensorList* dynQuantScaleOut = nullptr;

    int64_t splitItem = 3L; // 单单单：x/weight/y 均为单 tensor
    int64_t groupType = 2L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;

    std::vector<int8_t> xHostData(k * m / fp4PackRatio, 0x21);
    std::vector<int8_t> weightHostData(k * n / fp4PackRatio, 0x43);
    std::vector<int8_t> pertokenHostData(S * m * 2L, 129);
    std::vector<int8_t> scaleHostData(S * n * 2L, 130);
    std::vector<uint16_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<int64_t> groupListData;
    int64_t prefixSum = 0L;
    for (auto kG : kGroups) {
        prefixSum += kG;
        groupListData.push_back(prefixSum);
    }

    aclTensor* xTensor = nullptr;
    ret = CreateAclTensorWithLayout(xHostData, xViewShape, xStrides, xStorageShape, &xDeviceAddr,
                                    aclDataType::ACL_FLOAT4_E2M1, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    MakeSingleTensorList(xTensor, &x);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> xPtr(x, aclDestroyTensorList);

    aclTensor* weightTensor = nullptr;
    ret = CreateContiguousAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT4_E2M1,
                                    &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    MakeSingleTensorList(weightTensor, &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightPtr(weight, aclDestroyTensorList);

    aclTensor* pertokenTensor = nullptr;
    ret = CreateAclTensorWithLayout(pertokenHostData, pertokenViewShape, pertokenStrides, pertokenStorageShape,
                                    &pertokenDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &pertokenTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void*)> pertokenDeviceAddrPtr(pertokenDeviceAddr, aclrtFree);
    MakeSingleTensorList(pertokenTensor, &perTokenScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> perTokenScalePtr(perTokenScale,
                                                                                           aclDestroyTensorList);

    aclTensor* scaleTensor = nullptr;
    ret = CreateContiguousAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                    &scaleTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    MakeSingleTensorList(scaleTensor, &scale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> scalePtr(scale, aclDestroyTensorList);

    aclTensor* yTensor = nullptr;
    ret = CreateContiguousAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &yTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    MakeSingleTensorList(yTensor, &out);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outPtr(out, aclDestroyTensorList);

    ret = CreateContiguousAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr,
                                             aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

    // 调用 aclnnGroupedMatmulV5 第一段接口
    ret = aclnnGroupedMatmulV5GetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用 aclnnGroupedMatmulV5 第二段接口
    ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 将结果从 Device 拷贝到 Host，逐组打印前 printCount 个元素
    auto ySize = GetShapeSize(yShape);
    std::vector<uint16_t> resultData(ySize, 0);
    ret = aclrtMemcpy(resultData.data(), ySize * sizeof(resultData[0]), yDeviceAddr, ySize * sizeof(resultData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // 期望值（见文件头说明）：组 0 为 2048*a*b ∈ {1536,2048,3072,4096}，组 1 为 3072*a*b ∈ {2304,3072,4608,6144}
    for (int64_t g = 0; g < groupNum; g++) {
        for (int64_t j = 0; j < printCount; j++) {
            LOG_PRINT("y[%ld][%ld] (flat in group) is: %g\n", g, j, Bf16ToFloat(resultData[g * m * n + j]));
        }
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclnnGroupedMatmulMxFp4SplitKTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 MXFP4 split-K test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
