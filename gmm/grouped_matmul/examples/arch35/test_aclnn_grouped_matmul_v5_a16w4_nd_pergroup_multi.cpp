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
 * \file test_aclnn_grouped_matmul_v5_a16w4_nd_pergroup_multi.cpp
 * \brief aclnnGroupedMatmulV5 A16W4 ND pergroup 伪量化多多多场景调用示例：
 *        x 为多 tensor (BFLOAT16)，weight 为多 tensor 2D (INT4, ND)，
 *        antiquantScale 为多 tensor 2D (G_i, N_i) pergroup 模式，
 *        y 为多 tensor (BFLOAT16)，groupType=-1, splitItem=0。
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

class DeviceAddrListGuard {
public:
    explicit DeviceAddrListGuard(std::vector<void*>& addrs)
        : addrs_(addrs)
    {}
    ~DeviceAddrListGuard()
    {
        for (auto addr : addrs_) {
            if (addr != nullptr) {
                aclrtFree(addr);
            }
        }
    }
    DeviceAddrListGuard(const DeviceAddrListGuard&) = delete;
    DeviceAddrListGuard& operator=(const DeviceAddrListGuard&) = delete;

private:
    std::vector<void*>& addrs_;
};

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
int CreateAclTensorListMulti(const std::vector<std::vector<T>>& hostDataList,
                             const std::vector<std::vector<int64_t>>& shapes, std::vector<void*>& deviceAddrList,
                             aclDataType dataType, aclTensorList** tensor)
{
    int size = shapes.size();
    std::vector<aclTensor*> tensors(size, nullptr);
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

int aclnnGourpedMatmulTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // A16W4 ND pergroup 多多多场景参数：
    //   x: 多tensor，每个 (m_i, K), BFLOAT16, ND
    //   weight: 多tensor，每个 (K, N), INT4, ND
    //   antiquantScale: 多tensor，每个 (G_i, N), BFLOAT16, ND — pergroup模式（2D）
    //   bias: 多tensor，每个 (N,), BFLOAT16, ND
    //   y: 多tensor，每个 (m_i, N), BFLOAT16, ND
    //   groupSize = K / G_i = 32
    //   groupType = -1 (NO_SPLIT), splitItem = 0 (y多tensor)
    int64_t m0 = 32L;
    int64_t m1 = 32L;
    int64_t k = 128L;
    int64_t n = 64L;
    int64_t groupSize = 32L;
    int64_t g = k / groupSize; // pergroup数 G_i = 4

    std::vector<std::vector<int64_t>> xShapes = {{m0, k}, {m1, k}};
    std::vector<std::vector<int64_t>> weightShapes = {{k, n}, {k, n}};
    std::vector<std::vector<int64_t>> biasShapes = {{n}, {n}};
    std::vector<std::vector<int64_t>> antiquantScaleShapes = {{g, n}, {g, n}};
    std::vector<std::vector<int64_t>> antiquantOffsetShapes = {{g, n}, {g, n}};
    std::vector<std::vector<int64_t>> yShapes = {{m0, n}, {m1, n}};

    std::vector<void*> xDeviceAddrList;
    std::vector<void*> weightDeviceAddrList;
    std::vector<void*> biasDeviceAddrList;
    std::vector<void*> antiquantScaleDeviceAddrList;
    std::vector<void*> antiquantOffsetDeviceAddrList;
    std::vector<void*> yDeviceAddrList;

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

    int64_t splitItem = 0L;
    int64_t groupType = -1L; // NO_SPLIT
    int64_t groupListType = 0L;
    int64_t actType = 0L;

    // x: BFLOAT16, 多tensor
    std::vector<std::vector<int16_t>> xHostDataList;
    for (int64_t i = 0; i < 2; i++) {
        int64_t mi = (i == 0) ? m0 : m1;
        std::vector<int16_t> data(mi * k);
        for (int64_t j = 0; j < mi * k; j++) {
            data[j] = static_cast<int16_t>((j + i * 50) % 100 + 1);
        }
        xHostDataList.push_back(data);
    }
    ret = CreateAclTensorListMulti(xHostDataList, xShapes, xDeviceAddrList, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> xPtr(x, aclDestroyTensorList);
    DeviceAddrListGuard xDeviceAddrGuard(xDeviceAddrList);

    // weight: INT4, 多tensor, 每个element 4 bit, 用int8存储
    std::vector<std::vector<int8_t>> weightHostDataList;
    for (int64_t i = 0; i < 2; i++) {
        std::vector<int8_t> data(k * n / 2);
        for (int64_t j = 0; j < k * n / 2; j++) {
            data[j] = static_cast<int8_t>((j + i * 37) % 256);
        }
        weightHostDataList.push_back(data);
    }
    ret = CreateAclTensorListMulti(weightHostDataList, weightShapes, weightDeviceAddrList, aclDataType::ACL_INT4,
                                   &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightPtr(weight, aclDestroyTensorList);
    DeviceAddrListGuard weightDeviceAddrGuard(weightDeviceAddrList);

    // bias: FLOAT32, 多tensor (A16W4场景x为BF16时bias支持BFLOAT16/FLOAT32，通用通路默认期望FLOAT)
    std::vector<std::vector<float>> biasHostDataList;
    for (int64_t i = 0; i < 2; i++) {
        std::vector<float> data(n);
        for (int64_t j = 0; j < n; j++) {
            data[j] = j * 0.5f + 1.0f + i * 10.0f;
        }
        biasHostDataList.push_back(data);
    }
    ret = CreateAclTensorListMulti(biasHostDataList, biasShapes, biasDeviceAddrList, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> biasPtr(bias, aclDestroyTensorList);
    DeviceAddrListGuard biasDeviceAddrGuard(biasDeviceAddrList);

    // antiquantScale: BFLOAT16, pergroup 2D (G_i, N), 多tensor
    std::vector<std::vector<int16_t>> antiquantScaleHostDataList;
    for (int64_t i = 0; i < 2; i++) {
        std::vector<int16_t> data(g * n);
        for (int64_t j = 0; j < g * n; j++) {
            data[j] = static_cast<int16_t>((j + i * 3) % 5 + 1);
        }
        antiquantScaleHostDataList.push_back(data);
    }
    ret = CreateAclTensorListMulti(antiquantScaleHostDataList, antiquantScaleShapes, antiquantScaleDeviceAddrList,
                                   aclDataType::ACL_BF16, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> antiquantScalePtr(antiquantScale,
                                                                                            aclDestroyTensorList);
    DeviceAddrListGuard antiquantScaleDeviceAddrGuard(antiquantScaleDeviceAddrList);

    // antiquantOffset: BFLOAT16, pergroup 2D (G_i, N), 多tensor（多多多场景通用通路要求antiquantOffset非空）
    std::vector<std::vector<int16_t>> antiquantOffsetHostDataList;
    for (int64_t i = 0; i < 2; i++) {
        std::vector<int16_t> data(g * n);
        for (int64_t j = 0; j < g * n; j++) {
            data[j] = static_cast<int16_t>((j + i * 7) % 4);
        }
        antiquantOffsetHostDataList.push_back(data);
    }
    ret = CreateAclTensorListMulti(antiquantOffsetHostDataList, antiquantOffsetShapes, antiquantOffsetDeviceAddrList,
                                   aclDataType::ACL_BF16, &antiquantOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> antiquantOffsetPtr(antiquantOffset,
                                                                                             aclDestroyTensorList);
    DeviceAddrListGuard antiquantOffsetDeviceAddrGuard(antiquantOffsetDeviceAddrList);

    // y: BFLOAT16, 多tensor
    std::vector<std::vector<uint16_t>> yHostDataList = {std::vector<uint16_t>(m0 * n, 0),
                                                        std::vector<uint16_t>(m1 * n, 0)};
    ret = CreateAclTensorListMulti(yHostDataList, yShapes, yDeviceAddrList, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outPtr(out, aclDestroyTensorList);
    DeviceAddrListGuard yDeviceAddrGuard(yDeviceAddrList);

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
    for (size_t i = 0; i < yShapes.size(); i++) {
        auto size = GetShapeSize(yShapes[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddrList[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < 10 && j < size; j++) {
            LOG_PRINT("result[%zu][%ld] is: %d\n", i, j, resultData[j]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulV5 A16W4 pergroup multi test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
