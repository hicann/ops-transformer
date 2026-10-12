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
 * \file test_aclnn_dense_lightning_indexer_softmax_lse.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t s1 = 4096;
    int64_t s2 = 4096;
    int64_t n1Index = 8;
    int64_t n2Index = 1;
    int64_t dQueryIndex = 128;
    int64_t t1 = s1;
    int64_t t2 = s2;
    int64_t G = n1Index / n2Index;

    std::vector<int64_t> qIndexShape = {t1, n1Index, dQueryIndex};
    std::vector<int64_t> kIndexShape = {t2, n2Index, dQueryIndex};
    std::vector<int64_t> weightShape = {t1, n1Index};
    std::vector<int64_t> softmaxMaxIndexShape = {n2Index, t1};
    std::vector<int64_t> softmaxSumIndexShape = {n2Index, t1};

    void* qIndexDeviceAddr = nullptr;
    void* kIndexDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* softmaxMaxIndexDeviceAddr = nullptr;
    void* softmaxSumIndexDeviceAddr = nullptr;

    aclTensor* qIndex = nullptr;
    aclTensor* kIndex = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* softmaxMaxIndex = nullptr;
    aclTensor* softmaxSumIndex = nullptr;

    std::vector<aclFloat16> qIndexHostData(t1 * n1Index * dQueryIndex, aclFloatToFloat16(0.2));
    std::vector<aclFloat16> kIndexHostData(t2 * n2Index * dQueryIndex, aclFloatToFloat16(0.1));
    std::vector<aclFloat16> weightHostData(t1 * n1Index, aclFloatToFloat16(0.005));

    std::vector<float> softmaxMaxIndexHostData(t1 * n2Index, 25.4483f);
    std::vector<float> softmaxSumIndexHostData(t1 * n2Index, 1.0f);

    ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxIndexHostData, softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr,
                          aclDataType::ACL_FLOAT, &softmaxMaxIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumIndexHostData, softmaxSumIndexShape, &softmaxSumIndexDeviceAddr,
                          aclDataType::ACL_FLOAT, &softmaxSumIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> acSeqQLenOp = {t1};
    std::vector<int64_t> acSeqKvLenOp = {t2};
    aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
    aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t sparseMode = 3;

    char layOut[5] = {'T', 'N', 'D', 0};

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize第一段接口
    ret = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(qIndex, kIndex, weight, acSeqQLen, acSeqKvLen, layOut,
                                                               sparseMode, preTokens, nextTokens, softmaxMaxIndex,
                                                               softmaxSumIndex, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnDenseLightningIndexerSoftmaxLse第二段接口
    ret = aclnnDenseLightningIndexerSoftmaxLse(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLse failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr);
    PrintOutResult(softmaxSumIndexShape, &softmaxSumIndexDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(qIndex);
    aclDestroyTensor(kIndex);
    aclDestroyTensor(weight);
    aclDestroyTensor(softmaxMaxIndex);
    aclDestroyTensor(softmaxSumIndex);

    // 7. 释放device资源
    aclrtFree(qIndexDeviceAddr);
    aclrtFree(kIndexDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(softmaxMaxIndexDeviceAddr);
    aclrtFree(softmaxSumIndexDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
