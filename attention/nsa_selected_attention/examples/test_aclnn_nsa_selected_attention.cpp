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
 * \file test_aclnn_nsa_selected_attention.cpp
 * \brief
 */

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_selected_attention.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
void CopyOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
              aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtDestroyContext(context);
              aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *attentionOut, aclTensor *softmaxMax,
                  aclTensor *softmaxSum, void *qDeviceAddr, void *kDeviceAddr, void *vDeviceAddr,
                  void *attentionOutDeviceAddr, void *softmaxMaxDeviceAddr, void *softmaxSumDeviceAddr,
                  uint64_t workspaceSize, void *workspaceAddr, aclIntArray *actualSeqQLen, aclIntArray *actualSeqKvLen,
                  int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (attentionOut != nullptr) {
        aclDestroyTensor(attentionOut);
    }
    if (softmaxMax != nullptr) {
        aclDestroyTensor(softmaxMax);
    }
    if (softmaxSum != nullptr) {
        aclDestroyTensor(softmaxSum);
    }
    if (actualSeqQLen != nullptr) {
        aclDestroyIntArray(actualSeqQLen);
    }
    if (actualSeqKvLen != nullptr) {
        aclDestroyIntArray(actualSeqKvLen);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (attentionOutDeviceAddr != nullptr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (softmaxMaxDeviceAddr != nullptr) {
        aclrtFree(softmaxMaxDeviceAddr);
    }
    if (softmaxSumDeviceAddr != nullptr) {
        aclrtFree(softmaxSumDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
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
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_selected_attention 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行
    int64_t batch = 2;
    int64_t s1 = 512;
    int64_t s2 = 2048;
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 4;
    int64_t n2 = 4;
    std::vector<int64_t> qShape = {batch * s1, n2 * g, d1};
    std::vector<int64_t> kShape = {batch * s2, n2, d1};
    std::vector<int64_t> vShape = {batch * s2, n2, d2};
    std::vector<int64_t> topKIndicesShape = {batch * s1, n2, 16};
    std::vector<int64_t> attentionOutShape = {batch * s1, n2 * g, d2};
    std::vector<int64_t> softmaxMaxShape = {batch * s1, n2 * g, 8};
    std::vector<int64_t> softmaxSumShape = {batch * s1, n2 * g, 8};

    double scaleValue = 1.0;
    int64_t headNum = 16;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int64_t sparseMod = 2;
    char layOut[] = "TND";

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *topKIndicesDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *topKIndices = nullptr;
    aclTensor *attenMaskOptional = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;
    aclTensor *attentionOut = nullptr;

    std::vector<int64_t> actualSeqQLenVec = {512, 1024};
    std::vector<int64_t> actualSeqKvLenVec = {2048, 4096};
    aclIntArray *actualSeqQLenOptional = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
    aclIntArray *actualSeqKvLenOptional = aclCreateIntArray(actualSeqKvLenVec.data(), actualSeqKvLenVec.size());

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1);
    std::vector<aclFloat16> vHostData(GetShapeSize(vShape), 1);
    std::vector<int32_t> topkIndicesHostData(GetShapeSize(topKIndicesShape), 2);
    std::vector<float> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0);
    std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 0.0);
    std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 0.0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // 创建acl Tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(topkIndicesHostData, topKIndicesShape, &topKIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &topKIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnNsaSelectedAttention第一段接口
    ret = aclnnNsaSelectedAttentionGetWorkspaceSize(q, k, v, topKIndices, attenMaskOptional, actualSeqQLenOptional,
                                                    actualSeqKvLenOptional, scaleValue, headNum, layOut, sparseMod,
                                                    selectedBlockSize, selectedBlockCount, softmaxMax, softmaxSum,
                                                    attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); FreeResource(
                      q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                      attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                      actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
                  return ret);
    }

    // 调用aclnnNsaSelectedAttention第二段接口
    ret = aclnnNsaSelectedAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttention failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                           attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                           workspaceAddr, actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    CopyOutResult<float>(softmaxMaxShape, &softmaxMaxDeviceAddr);
    CopyOutResult<float>(softmaxSumShape, &softmaxSumDeviceAddr);
    CopyOutResult<aclFloat16>(attentionOutShape, &attentionOutDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                 attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                 actualSeqQLenOptional, actualSeqKvLenOptional, deviceId, &context, &stream);

    return 0;
}
