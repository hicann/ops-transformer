/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_matmul_swiglu_quant.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t E = 4;
    int64_t M = 8192;
    int64_t N = 4096;
    int64_t K = 7168;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShape = {E, N / 32, K / 16, 16, 32};
    std::vector<int64_t> weightScaleShape = {E, N};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<int8_t> weightHostData(E * N * K, 0);
    std::vector<float> weightScaleHostData(E * N, 0);
    std::vector<float> xScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData(E, 0);
    std::vector<int8_t> outputHostData(M * N / 2, 0);
    std::vector<float> outputScaleHostData(M, 0);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8,
                          aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightScale aclTensor
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          aclFormat::ACL_FORMAT_ND, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xScale aclTensor
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                          aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8,
                          aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outputScale aclTensor
    ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          aclFormat::ACL_FORMAT_ND, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulSwigluQuant第一段接口
    ret = aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(x, weight, nullptr, nullptr, weightScale, xScale, groupList,
                                                        output, outputScale, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulSwigluQuant第二段接口
    ret = aclnnGroupedMatmulSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 10; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, out1Data[j]);
    }
    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 10; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, out2Data[j]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightScaleDeviceAddr);
    aclrtFree(xScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
