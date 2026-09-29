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
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ring_attention_update.h"

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

int main()
{
    // 1. (固定写法)device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t batchNum = 1;
    int64_t headNum = 1;
    int64_t seqSize = 2;
    int64_t headDim = 64;
    int64_t headSize = headNum * headDim;

    std::vector<int64_t> prevAttnOutShape = {seqSize, batchNum, headSize};
    std::vector<int64_t> prevSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
    std::vector<int64_t> prevSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
    std::vector<int64_t> curAttnOutShape = {seqSize, batchNum, headSize};
    std::vector<int64_t> curSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
    std::vector<int64_t> curSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
    std::vector<int64_t> actualSeqQlenOptionalShape = {seqSize};

    std::vector<int64_t> attnOutShape = {seqSize, batchNum, headSize};
    std::vector<int64_t> softmaxMaxShape = {batchNum * seqSize, headNum, 8};
    std::vector<int64_t> softmaxSumShape = {batchNum * seqSize, headNum, 8};

    void *prevAttnOutDeviceAddr = nullptr;
    void *prevSoftmaxMaxDeviceAddr = nullptr;
    void *prevSoftmaxSumDeviceAddr = nullptr;
    void *curAttnOutDeviceAddr = nullptr;
    void *curSoftmaxMaxDeviceAddr = nullptr;
    void *curSoftmaxSumDeviceAddr = nullptr;
    void *actualSeqQlenOptionalDeviceAddr = nullptr;

    void *attnOutDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;

    aclTensor *prevAttnOut = nullptr;
    aclTensor *prevSoftmaxMax = nullptr;
    aclTensor *prevSoftmaxSum = nullptr;
    aclTensor *curAttnOut = nullptr;
    aclTensor *curSoftmaxMax = nullptr;
    aclTensor *curSoftmaxSum = nullptr;
    aclTensor *actualSeqQlenOptional = nullptr;

    aclTensor *attnOut = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;

    std::vector<float> prevAttnOutHostData(seqSize * batchNum * headSize, 1);
    std::vector<float> prevSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
    std::vector<float> prevSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
    std::vector<float> curAttnOutHostData(seqSize * batchNum * headSize, 1);
    std::vector<float> curSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
    std::vector<float> curSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
    std::vector<int64_t> actualSeqQlenOptionalHostData = {0, seqSize};

    std::vector<float> attnOutHostData(seqSize * batchNum * headSize, 1);
    std::vector<float> softmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
    std::vector<float> softmaxSumHostData(batchNum * headNum * seqSize * 8, 1);

    char *inputLayoutOptional = "TND";
    char *inputSoftmaxLayoutOptional = "TND";
    // 创建prevAttnOut aclTensor
    ret = CreateAclTensor(prevAttnOutHostData, prevAttnOutShape, &prevAttnOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &prevAttnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建prevSoftmaxMax aclTensor
    ret = CreateAclTensor(prevSoftmaxMaxHostData, prevSoftmaxMaxShape, &prevSoftmaxMaxDeviceAddr,
                          aclDataType::ACL_FLOAT, &prevSoftmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建prevSoftmaxSum aclTensor
    ret = CreateAclTensor(prevSoftmaxSumHostData, prevSoftmaxSumShape, &prevSoftmaxSumDeviceAddr,
                          aclDataType::ACL_FLOAT, &prevSoftmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建curAttnOut aclTensor
    ret = CreateAclTensor(curAttnOutHostData, curAttnOutShape, &curAttnOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &curAttnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建curSoftmaxMax aclTensor
    ret = CreateAclTensor(curSoftmaxMaxHostData, curSoftmaxMaxShape, &curSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &curSoftmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建curSoftmaxSum aclTensor
    ret = CreateAclTensor(curSoftmaxSumHostData, curSoftmaxSumShape, &curSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &curSoftmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建actualSeqQlenOptional aclTensor
    ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr,
                          aclDataType::ACL_INT64, &actualSeqQlenOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建attnOut aclTensor
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建softmaxMax aclTensor
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建softmaxSum aclTensor
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnRingAttentionUpdateV2第一段接口
    ret = aclnnRingAttentionUpdateV2GetWorkspaceSize(
        prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQlenOptional,
        inputLayoutOptional, inputSoftmaxLayoutOptional, attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnRingAttentionUpdateV2第二段接口
    ret = aclnnRingAttentionUpdateV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto attnOutSize = GetShapeSize(attnOutShape);
    std::vector<float> attnOutResultData(attnOutSize, 0);
    ret = aclrtMemcpy(attnOutResultData.data(), attnOutResultData.size() * sizeof(attnOutResultData[0]),
                      attnOutDeviceAddr, attnOutSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < attnOutSize; i++) {
        LOG_PRINT("attnOutResultData[%ld] is: %f\n", i, attnOutResultData[i]);
    }

    auto softmaxMaxSize = GetShapeSize(softmaxMaxShape);
    std::vector<float> softmaxMaxResultData(softmaxMaxSize, 0);
    ret = aclrtMemcpy(softmaxMaxResultData.data(), softmaxMaxResultData.size() * sizeof(softmaxMaxResultData[0]),
                      softmaxMaxDeviceAddr, softmaxMaxSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < softmaxMaxSize; i++) {
        LOG_PRINT("softmaxMaxResultData[%ld] is: %f\n", i, softmaxMaxResultData[i]);
    }

    auto softmaxSumSize = GetShapeSize(softmaxSumShape);
    std::vector<float> softmaxSumResultData(softmaxSumSize, 0);
    ret = aclrtMemcpy(softmaxSumResultData.data(), softmaxSumResultData.size() * sizeof(softmaxSumResultData[0]),
                      softmaxSumDeviceAddr, softmaxSumSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < softmaxSumSize; i++) {
        LOG_PRINT("softmaxSumResultData[%ld] is: %f\n", i, softmaxSumResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(prevAttnOut);
    aclDestroyTensor(prevSoftmaxMax);
    aclDestroyTensor(prevSoftmaxSum);
    aclDestroyTensor(curAttnOut);
    aclDestroyTensor(curSoftmaxMax);
    aclDestroyTensor(curSoftmaxSum);
    aclDestroyTensor(actualSeqQlenOptional);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(prevAttnOutDeviceAddr);
    aclrtFree(prevSoftmaxMaxDeviceAddr);
    aclrtFree(prevSoftmaxSumDeviceAddr);
    aclrtFree(curAttnOutDeviceAddr);
    aclrtFree(curSoftmaxMaxDeviceAddr);
    aclrtFree(curSoftmaxSumDeviceAddr);
    aclrtFree(actualSeqQlenOptionalDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
