/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclFormat FormatType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, FormatType, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, ACL_FORMAT_ND, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

template <typename T1, typename T2>
auto Ceil(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGroupedMatmulSwigluQuantV2Test(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t E = 8;
    int64_t M = 2048;
    int64_t N = 4096;
    int64_t K = 7168;
    constexpr int64_t PRINT_NUM = 10;

    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShape = {E, K, N};
    std::vector<int64_t> weightScaleShape = {E, N};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *outputScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *output = nullptr;
    aclTensor *outputScale = nullptr;
    aclTensorList *weightAssistMatri = nullptr;
    aclTensorList *smoothScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 1);
    std::vector<int8_t> weightHostData(E * N * K, 1);
    std::vector<uint16_t> weightScaleHostData(E * N, 0x3F80); // BF16 representation of 1.0
    std::vector<float> xScaleHostData(M, 1.0F);
    std::vector<int64_t> groupListHostData(E, 1);
    std::vector<int8_t> outputHostData(M * N / 2, 1);
    std::vector<float> outputScaleHostData(M, 1.0F);
    std::vector<int64_t> tuningConfigData = {1};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> tuningConfigPtr(tuningConfig,
                                                                                       aclDestroyIntArray);

    int64_t quantMode = 0;    // 最后一次量化的类型
    int64_t dequantMode = 0;  // gmm计算时的量化类型
    int64_t dequantDtype = 0; // 0 表示fp32
    int64_t groupListType = 1;

    // 创建x aclTensor
    std::vector<unsigned char> xHostDataUnsigned(xHostData.begin(), xHostData.end());
    ret = CreateAclTensor<uint8_t>(xHostDataUnsigned, xShape, &xDeviceAddr, aclDataType::ACL_INT8,
                                   aclFormat::ACL_FORMAT_ND, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensorList
    std::vector<std::vector<int8_t>> weightHostDataList = {weightHostData};
    std::vector<std::vector<int64_t>> weightShapeList = {weightShape};
    ret = CreateAclTensorList<int8_t>(weightHostDataList, weightShapeList, &weightDeviceAddr, aclDataType::ACL_INT8,
                                      &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                               aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建weightScale aclTensorList
    std::vector<std::vector<uint16_t>> weightScaleHostDataList = {weightScaleHostData};
    std::vector<std::vector<int64_t>> weightScaleShapeList = {weightScaleShape};
    ret = CreateAclTensorList<uint16_t>(weightScaleHostDataList, weightScaleShapeList, &weightScaleDeviceAddr,
                                        aclDataType::ACL_BF16, &weightScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xScale aclTensor
    ret = CreateAclTensor<float>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建group_list aclTensor
    ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   aclFormat::ACL_FORMAT_ND, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y aclTensor
    ret = CreateAclTensor<int8_t>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8,
                                  aclFormat::ACL_FORMAT_ND, &output);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建yScale aclTensor
    ret = CreateAclTensor<float>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &outputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulSwigluQuantV2第一段接口
    ret = aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
        x, weight, weightScale, nullptr, nullptr, xScale, nullptr, groupList, dequantMode, dequantDtype, quantMode,
        groupListType, nullptr, output, outputScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulSwigluQuantV2第二段接口
    ret = aclnnGroupedMatmulSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> outputData(size, 0);
    ret = aclrtMemcpy(outputData.data(), size * sizeof(outputData[0]), outputDeviceAddr, size * sizeof(outputData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t j = 0; j < size && j < PRINT_NUM; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, outputData[j]);
    }

    size = GetShapeSize(outputScaleShape);
    std::vector<float> outputScaleData(size, 0.0F);
    ret = aclrtMemcpy(outputScaleData.data(), size * sizeof(outputScaleData[0]), outputScaleDeviceAddr,
                      size * sizeof(outputScaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScaleData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t j = 0; j < size && j < PRINT_NUM; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, outputScaleData[j]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedMatmulSwigluQuantV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2Test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
