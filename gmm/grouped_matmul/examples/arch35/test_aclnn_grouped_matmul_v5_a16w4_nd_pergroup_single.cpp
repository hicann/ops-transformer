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
 * \file test_aclnn_grouped_matmul_v5_a16w4_nd_pergroup_single.cpp
 * \brief aclnnGroupedMatmulV5 A16W4 ND pergroup 伪量化单单单场景调用示例：
 *        x 为单 tensor (BFLOAT16)，weight 为单 tensor 3D (INT4, ND)，
 *        antiquantScale 为单 tensor 3D (E, G, N) pergroup 模式，
 *        y 为单 tensor (BFLOAT16)，groupType=0, splitItem=3。
 */

#include <iostream>
#include <vector>
#include <memory>
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
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
int CreateAclTensorList(const std::vector<T>& hostData, const std::vector<std::vector<int64_t>>& shapes,
                        void** deviceAddr, aclDataType dataType, aclTensorList** tensor)
{
    int size = shapes.size();
    std::vector<aclTensor*> tensors(size, nullptr);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostData, shapes[i], deviceAddr + i, dataType, &tensors[i]);
        if (ret != ACL_SUCCESS) {
            for (int j = 0; j < i; j++) {
                aclDestroyTensor(tensors[j]);
                aclrtFree(*(deviceAddr + j));
                *(deviceAddr + j) = nullptr;
            }
            return ret;
        }
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int aclnnGourpedMatmulTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // A16W4 ND pergroup 单单单场景参数
    int64_t m = 64L;
    int64_t k = 128L;
    int64_t n = 64L;
    int64_t e = 2L;
    int64_t groupSize = 32L;
    int64_t g = k / groupSize; // pergroup数 G = 4

    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShape = {{e, k, n}};
    std::vector<std::vector<int64_t>> biasShape = {{e, n}};
    std::vector<std::vector<int64_t>> antiquantScaleShape = {{e, g, n}};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {{e}};
    std::vector<int64_t> groupListData = {m / e, m};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* antiquantScaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;

    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclTensor* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* perTokenScale = nullptr;
    aclTensorList* activationInput = nullptr;
    aclTensorList* activationQuantScale = nullptr;
    aclTensorList* activationQuantOffset = nullptr;
    aclTensorList* out = nullptr;
    aclTensorList* activationFeatureOut = nullptr;
    aclTensorList* dynQuantScaleOut = nullptr;

    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;

    // x: BFLOAT16
    std::vector<int16_t> xHostData(m * k);
    for (int64_t i = 0; i < m * k; i++) {
        xHostData[i] = static_cast<int16_t>(i % 100 + 1);
    }
    ret = CreateAclTensorList(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> xPtr(x, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);

    // weight: INT4, 每个element 4 bit, 用int8存储, 2 elements per byte
    std::vector<int8_t> weightHostData(e * k * n / 2);
    for (int64_t i = 0; i < e * k * n / 2; i++) {
        weightHostData[i] = static_cast<int8_t>(i % 256);
    }
    ret = CreateAclTensorList(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightPtr(weight, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);

    // bias: FLOAT32 (A16W4场景x为BF16时bias支持BFLOAT16/FLOAT32，通用通路默认期望FLOAT)
    std::vector<float> biasHostData(e * n);
    for (int64_t i = 0; i < e * n; i++) {
        biasHostData[i] = i * 0.5f + 1.0f;
    }
    ret = CreateAclTensorList(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> biasPtr(bias, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);

    // antiquantScale: BFLOAT16, pergroup 3D (E, G, N)
    std::vector<int16_t> antiquantScaleHostData(e * g * n);
    for (int64_t i = 0; i < e * g * n; i++) {
        antiquantScaleHostData[i] = static_cast<int16_t>(i % 5 + 1);
    }
    ret = CreateAclTensorList(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                              aclDataType::ACL_BF16, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> antiquantScalePtr(antiquantScale,
                                                                                            aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);

    // y: BFLOAT16
    std::vector<uint16_t> yHostData(m * n, 0);
    ret = CreateAclTensorList(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outPtr(out, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);

    // groupList: INT64
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

    // 调用aclnnGroupedMatmulV5第一段接口
    ret = aclnnGroupedMatmulV5GetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n[ERROR msg]%s\n",
                                            ret, aclGetRecentErrMsg());
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnGroupedMatmulV5第二段接口
    ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出结果
    auto size = GetShapeSize(yShape[0]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 10 && j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulV5 A16W4 pergroup single test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
