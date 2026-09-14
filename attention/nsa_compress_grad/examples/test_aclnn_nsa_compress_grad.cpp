/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_nsa_compress_grad.cpp
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_grad.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

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

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，acl初始化
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **xOrResult)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续xOrResult的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                 shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FreeResource(aclTensor *outputGrad, aclTensor *inputKV, aclTensor *weight, aclTensor *inputGradOut,
                  aclTensor *weightGradOut, aclIntArray *actSeqLen, void *outputGradDeviceAddr, void *inputKVDeviceAddr,
                  void *weightDeviceAddr, void *inputGradOutDeviceAddr, void *weightGradOutDeviceAddr,
                  uint64_t workspaceSize, void *workspaceAddr, int32_t deviceId, aclrtContext context,
                  aclrtStream stream)
{
    if (outputGrad != nullptr) {
        aclDestroyTensor(outputGrad);
    }
    if (inputKV != nullptr) {
        aclDestroyTensor(inputKV);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (inputGradOut != nullptr) {
        aclDestroyTensor(inputGradOut);
    }
    if (weightGradOut != nullptr) {
        aclDestroyTensor(weightGradOut);
    }
    if (actSeqLen != nullptr) {
        aclDestroyIntArray(actSeqLen);
    }

    if (outputGradDeviceAddr != nullptr) {
        aclrtFree(outputGradDeviceAddr);
    }
    if (inputKVDeviceAddr != nullptr) {
        aclrtFree(inputKVDeviceAddr);
    }
    if (weightDeviceAddr != nullptr) {
        aclrtFree(weightDeviceAddr);
    }
    if (inputGradOutDeviceAddr != nullptr) {
        aclrtFree(inputGradOutDeviceAddr);
    }
    if (weightGradOutDeviceAddr != nullptr) {
        aclrtFree(weightGradOutDeviceAddr);
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
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
    // 1. （固定写法）device/context/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t headNum = 64;
    int64_t headDim = 128;
    int64_t blockSize = 32;
    int64_t blockStride = 16;
    int64_t blockNum = 15;
    int64_t seqLensSum = 272;
    int64_t seqLen = 3;
    std::vector<int64_t> outputGradShape = {blockNum, headNum, headDim};
    std::vector<int64_t> inputKVShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightShape = {blockSize, headNum};
    std::vector<int64_t> inputGradOutShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightGradOutShape = {blockSize, headNum};
    int64_t SeqLenType = 0;
    char layOut[] = "TND";

    void *outputGradDeviceAddr = nullptr;
    void *inputKVDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *inputGradOutDeviceAddr = nullptr;
    void *weightGradOutDeviceAddr = nullptr;

    aclTensor *outputGrad = nullptr;
    aclTensor *inputKV = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *inputGradOut = nullptr;
    aclTensor *weightGradOut = nullptr;

    std::vector<float> inputGradOutHostData(seqLensSum * headNum * headDim, 0.0);
    std::vector<float> weightGradOutHostData(blockSize * headNum, 0.0);

    std::vector<float> outputGradHostData(blockNum * headNum * headDim, 1.0);
    std::vector<float> inputKVHostData(seqLensSum * headNum * headDim, 1.0);
    std::vector<float> weightHostData(blockSize * headNum, 1.0);
    std::vector<int64_t> actSeqLenOptionalHostData = {0, 128, 272};

    aclIntArray *actSeqLenOptional =
        aclCreateIntArray(actSeqLenOptionalHostData.data(), actSeqLenOptionalHostData.size());
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    auto freeResource = [&]() {
        FreeResource(outputGrad, inputKV, weight, inputGradOut, weightGradOut, actSeqLenOptional, outputGradDeviceAddr,
                     inputKVDeviceAddr, weightDeviceAddr, inputGradOutDeviceAddr, weightGradOutDeviceAddr,
                     workspaceSize, workspaceAddr, deviceId, context, stream);
    };

    // 创建dy aclTensor
    ret = CreateAclTensor(outputGradHostData, outputGradShape, &outputGradDeviceAddr, aclDataType::ACL_FLOAT16,
                          &outputGrad);
    CHECK_RET(ret == ACL_SUCCESS, freeResource(); return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(inputKVHostData, inputKVShape, &inputKVDeviceAddr, aclDataType::ACL_FLOAT16, &inputKV);
    CHECK_RET(ret == ACL_SUCCESS, freeResource(); return ret);
    // 创建gelu aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, freeResource(); return ret);

    ret = CreateAclTensor(inputGradOutHostData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, freeResource(); return ret);

    ret = CreateAclTensor(weightGradOutHostData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, freeResource(); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;
    // 调用aclnnNsaCompressGrad第一段接口
    ret = aclnnNsaCompressGradGetWorkspaceSize(outputGrad, inputKV, weight, actSeqLenOptional, blockSize, blockStride,
                                               SeqLenType, layOut, inputGradOut, weightGradOut, &workspaceSize,
                                               &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              freeResource(); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); freeResource();
                  return ret);
    }
    // 调用aclnnNsaCompressGrad第二段接口
    ret = aclnnNsaCompressGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGrad failed. ERROR: %d\n", ret); freeResource();
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); freeResource();
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); freeResource();
              return ret);
    // 6. 释放aclTensor、aclIntArray和device资源
    freeResource();
    return 0;
}
