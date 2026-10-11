/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_quant.h"
#include <iostream>
#include <vector>
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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* rowIdxDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expandedExpertIdxOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rowIdx = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 3;
    double scale = 1.0;
    double offset = 1.0;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expandedExpertIdxOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<int> rowIdxHostData = {0, 3, 1, 4, 2, 5};
    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expandedExpertIdxOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rowIdxHostData, idxShape, &rowIdxDeviceAddr, aclDataType::ACL_INT32, &rowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8,
                          &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32,
                          &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedExpertIdxOutHostData, idxOutShape, &expandedExpertIdxOutDeviceAddr,
                          aclDataType::ACL_INT32, &expandedExpertIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRoutingQuant第一段接口
    ret = aclnnMoeInitRoutingQuantGetWorkspaceSize(x, rowIdx, expertIdx, activeNum, scale, offset, expandedXOut,
                                                   expandedRowIdxOut, expandedExpertIdxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMoeInitRoutingQuant第二段接口
    ret = aclnnMoeInitRoutingQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuant failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr,
                      expandedXSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]),
                      expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expandedExpertIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedExpertIdxData(expandedExpertIdxSize, 0);
    ret =
        aclrtMemcpy(expandedExpertIdxData.data(), expandedExpertIdxData.size() * sizeof(expandedExpertIdxData[0]),
                    expandedExpertIdxOutDeviceAddr, expandedExpertIdxSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedExpertIdxSize; i++) {
        LOG_PRINT("expandedExpertIdxData[%ld] is: %d\n", i, expandedExpertIdxData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(rowIdx);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expandedExpertIdxOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(rowIdxDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expandedExpertIdxOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
