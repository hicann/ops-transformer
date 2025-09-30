/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
                    aclDataType dataType, aclTensor **tensor)
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

template <typename T>
int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
{
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    size *= sizeof(T);
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

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

  int main() {
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 8;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 1;
    int64_t batch = 8;
    int64_t bsdp = 1;
    int64_t dtype = 0;
    float shareInputWeight = 1.0;
    int64_t shareInputOffest = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> wShape = {e, k, n / 8};
    std::vector<int64_t> scaleShape = {e, 1, n};
    std::vector<int64_t> biasShape = {e, n};
    std::vector<int64_t> offsetShape = {e, 1, n};
    std::vector<int64_t> pertokenScaleShape = {m};
    std::vector<int64_t> groupListShape = {e};
    std::vector<int64_t> sharedInputShape = {bsdp, n};
    std::vector<int64_t> logitShape = {m};
    std::vector<int64_t> rowIndexShape = {m};
    std::vector<int64_t> outShape = {batch, n};
    std::vector<int64_t> tuningConfigVal = { 1 };

    void *xDeviceAddr = nullptr;
    void *wDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *sharedInputDeviceAddr = nullptr;
    void *logitDeviceAddr = nullptr;
    void *rowIndexDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* w = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* pertokenScale = nullptr;
    aclTensor* sharedInput = nullptr;
    aclTensor* logit = nullptr;
    aclTensor* rowIndex = nullptr;
    aclTensor* out = nullptr;

    std::vector<int8_t> xHostData(GetShapeSize(xShape));
    std::vector<int32_t> wHostData(GetShapeSize(wShape));
    std::vector<int64_t> scaleHostData(GetShapeSize(scaleShape));
    std::vector<float> biasHostData(GetShapeSize(biasShape));
    std::vector<float> offsetHostData(GetShapeSize(offsetShape));
    std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
    std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
    std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
    std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
    std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
    std::vector<float> outHostData(GetShapeSize(outShape));  // 实际上是float16半精度方式
    // 对groupList赋值
    groupListHostData[0] = 8;
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建int32_t 的w aclTensor，后续转为int_4
    ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT32, &w);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_INT64, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建pertokenScale aclTensor
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sharedInput aclTensor
    ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logit aclTensor
    ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rowIndex aclTensor
    ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    CHECK_RET(tuningConfig == nullptr, -1);
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    // 调用aclnnGroupedMatmulFinalizeRoutingV3第一段接口
    workspaceSize = 0;
    ret = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(x, w, scale, bias, offset, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, shareInputOffest, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulFinalizeRoutingV3第二段接口
    ret = aclnnGroupedMatmulFinalizeRoutingV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(w);
    aclDestroyTensor(scale);
    aclDestroyTensor(bias);
    aclDestroyTensor(offset);
    aclDestroyTensor(pertokenScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(sharedInput);
    aclDestroyTensor(logit);
    aclDestroyTensor(rowIndex);
    aclDestroyTensor(out);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(wDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(pertokenScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(sharedInputDeviceAddr);
    aclrtFree(logitDeviceAddr);
    aclrtFree(rowIndexDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyIntArray(tuningConfig);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}