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
 * \file test_norm_rope_concat.cpp
 * \brief
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
#include "aclnnop/aclnn_norm_rope_concat.h"

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

bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
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

    std::filebuf* buf = file.rdbuf();
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
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** result)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续result的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *result = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
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
    uint32_t batchSize = 1;
    uint32_t headNum = 10;
    uint32_t headDim = 6;
    uint32_t querySeq = 5;
    uint32_t keySeq = 5;
    uint32_t valueSeq = 5;
    uint32_t encoderQuerySeq = 3;
    uint32_t encoderKeySeq = 3;
    uint32_t encoderValueSeq = 3;
    uint32_t ropeSeq = 5;
    uint32_t ropeDim = 6;
    float eps = 1e-5f;
    uint32_t normType = 2;
    uint32_t normAddedType = 2;
    uint32_t ropeType = 1;
    uint32_t concatOrder = 0;

    std::vector<int64_t> queryShape = {batchSize, querySeq, headNum, headDim};
    std::vector<int64_t> keyShape = {batchSize, keySeq, headNum, headDim};
    std::vector<int64_t> valueShape = {batchSize, keySeq, headNum, headDim};
    std::vector<int64_t> encoderQueryShape = {batchSize, encoderQuerySeq, headNum, headDim};
    std::vector<int64_t> encoderKeyShape = {batchSize, encoderKeySeq, headNum, headDim};
    std::vector<int64_t> encoderValueShape = {batchSize, encoderKeySeq, headNum, headDim};

    std::vector<int64_t> normQueryWeightShape = {headDim};
    std::vector<int64_t> normQueryBiasShape = {headDim};
    std::vector<int64_t> normKeyWeightShape = {headDim};
    std::vector<int64_t> normKeyBiasShape = {headDim};
    std::vector<int64_t> normAddedQueryWeightShape = {headDim};
    std::vector<int64_t> normAddedQueryBiasShape = {headDim};
    std::vector<int64_t> normAddedKeyWeightShape = {headDim};
    std::vector<int64_t> normAddedKeyBiasShape = {headDim};

    std::vector<int64_t> ropeSinShape = {ropeSeq, headDim};
    std::vector<int64_t> ropeCosShape = {ropeSeq, headDim};

    std::vector<int64_t> queryOutputShape = {batchSize, headNum, querySeq * 2, headDim};
    std::vector<int64_t> keyOutputShape = {batchSize, headNum, querySeq * 2, headDim};
    std::vector<int64_t> valueOutputShape = {batchSize, headNum, querySeq * 2, headDim};

    std::vector<int64_t> normQueryMeanShape = {batchSize, querySeq, headNum, 1};
    std::vector<int64_t> normQueryRstdShape = {batchSize, querySeq, headNum, 1};
    std::vector<int64_t> normKeyMeanShape = {batchSize, keySeq, headNum, 1};
    std::vector<int64_t> normKeyRstdShape = {batchSize, keySeq, headNum, 1};
    std::vector<int64_t> normAddedQueryMeanShape = {batchSize, encoderQuerySeq, headNum, 1};
    std::vector<int64_t> normAddedQueryRstdShape = {batchSize, encoderQuerySeq, headNum, 1};
    std::vector<int64_t> normAddedKeyMeanShape = {batchSize, encoderKeySeq, headNum, 1};
    std::vector<int64_t> normAddedKeyRstdShape = {batchSize, encoderKeySeq, headNum, 1};

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* encoderQueryDeviceAddr = nullptr;
    void* encoderKeyDeviceAddr = nullptr;
    void* encoderValueDeviceAddr = nullptr;
    // layernorm
    void* normQueryWeightDeviceAddr = nullptr;
    void* normQueryBiasDeviceAddr = nullptr;
    void* normKeyWeightDeviceAddr = nullptr;
    void* normKeyBiasDeviceAddr = nullptr;
    void* normAddedQueryWeightDeviceAddr = nullptr;
    void* normAddedQueryBiasDeviceAddr = nullptr;
    void* normAddedKeyWeightDeviceAddr = nullptr;
    void* normAddedKeyBiasDeviceAddr = nullptr;
    // rope
    void* ropeSinDeviceAddr = nullptr;
    void* ropeCosDeviceAddr = nullptr;

    void* queryOutputDeviceAddr = nullptr;
    void* keyOutputDeviceAddr = nullptr;
    void* valueOutputDeviceAddr = nullptr;

    void* normQueryMeanDeviceAddr = nullptr;
    void* normQueryRstdDeviceAddr = nullptr;
    void* normKeyMeanDeviceAddr = nullptr;
    void* normKeyRstdDeviceAddr = nullptr;
    void* normAddedQueryMeanDeviceAddr = nullptr;
    void* normAddedQueryRstdDeviceAddr = nullptr;
    void* normAddedKeyMeanDeviceAddr = nullptr;
    void* normAddedKeyRstdDeviceAddr = nullptr;

    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* encoderQuery = nullptr;
    aclTensor* encoderKey = nullptr;
    aclTensor* encoderValue = nullptr;
    aclTensor* normQueryWeight = nullptr;
    aclTensor* normQueryBias = nullptr;
    aclTensor* normKeyWeight = nullptr;
    aclTensor* normKeyBias = nullptr;
    aclTensor* normAddedQueryWeight = nullptr;
    aclTensor* normAddedQueryBias = nullptr;
    aclTensor* normAddedKeyWeight = nullptr;
    aclTensor* normAddedKeyBias = nullptr;

    aclTensor* ropeSin = nullptr;
    aclTensor* ropeCos = nullptr;

    aclTensor* normQueryMean = nullptr;
    aclTensor* normQueryRstd = nullptr;
    aclTensor* normKeyMean = nullptr;
    aclTensor* normKeyRstd = nullptr;
    aclTensor* normAddedQueryMean = nullptr;
    aclTensor* normAddedQueryRstd = nullptr;
    aclTensor* normAddedKeyMean = nullptr;
    aclTensor* normAddedKeyRstd = nullptr;

    aclTensor* queryOutput = nullptr;
    aclTensor* keyOutput = nullptr;
    aclTensor* valueOutput = nullptr;

    std::vector<float> queryOutputHostData(batchSize * headNum * (querySeq * 2) + encoderQuerySeq * headDim, 0.0);
    std::vector<float> keyOutputHostData(batchSize * headNum * keySeq + encoderKeySeq * headDim, 0.0);
    std::vector<float> valueOutputHostData(batchSize * headNum * valueSeq + encoderValueSeq * headDim, 0.0);

    std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 4.0);
    std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 5.0);
    std::vector<float> encoderValueHostData(batchSize * headNum * encoderKeySeq * headDim, 6.0);

    std::vector<float> normQueryWeightHostData(headDim, 1.0);
    std::vector<float> normQueryBiasHostData(headDim, 2.0);
    std::vector<float> normKeyWeightHostData(headDim, 3.0);
    std::vector<float> normKeyBiasHostData(headDim, 4.0);
    std::vector<float> normAddedQueryWeightHostData(headDim, 5.0);
    std::vector<float> normAddedQueryBiasHostData(headDim, 6.0);
    std::vector<float> normAddedKeyWeightHostData(headDim, 7.0);
    std::vector<float> normAddedKeyBiasHostData(headDim, 8.0);
    std::vector<float> ropeSinHostData(ropeSeq * headDim, 9.0);
    std::vector<float> ropeCosHostData(ropeSeq * headDim, 10.0);

    std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 0.0);
    std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 0.0);
    std::vector<float> valueHostData(batchSize * headNum * keySeq * headDim, 0.0);
    std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
    std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 0.0);
    std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
    std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 0.0);
    std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
    std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
    std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
    std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);

    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &encoderQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT,
                          &encoderKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(encoderValueHostData, encoderValueShape, &encoderValueDeviceAddr, aclDataType::ACL_FLOAT,
                          &encoderValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normQueryBiasHostData, normQueryBiasShape, &normQueryBiasDeviceAddr, aclDataType::ACL_FLOAT,
                          &normQueryBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                          &normKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normKeyBiasHostData, normKeyBiasShape, &normKeyBiasDeviceAddr, aclDataType::ACL_FLOAT,
                          &normKeyBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedQueryBiasHostData, normAddedQueryBiasShape, &normAddedQueryBiasDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedQueryBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedKeyBiasHostData, normAddedKeyBiasShape, &normAddedKeyBiasDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedKeyBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryOutputHostData, queryOutputShape, &queryOutputDeviceAddr, aclDataType::ACL_FLOAT,
                          &queryOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyOutputHostData, keyOutputShape, &keyOutputDeviceAddr, aclDataType::ACL_FLOAT, &keyOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueOutputHostData, valueOutputShape, &valueOutputDeviceAddr, aclDataType::ACL_FLOAT,
                          &valueOutput);
    ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT,
                          &normQueryMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT,
                          &normQueryRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                          &normKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT,
                          &normKeyMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT,
                          &normKeyRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedQueryMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedQueryRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedKeyMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr,
                          aclDataType::ACL_FLOAT, &normAddedKeyRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    bool isTraining = false;
    // 调用aclnnGeGluBackward第一段接口
    ret = aclnnNormRopeConcatGetWorkspaceSize(
        query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
        normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin, ropeCos,
        normType, normAddedType, ropeType, concatOrder, eps, isTraining, queryOutput, keyOutput, valueOutput,
        normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
        normAddedKeyMean, normAddedKeyRstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGeGluBackward第二段接口
    ret = aclnnNormRopeConcat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcat failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(queryOutputShape);
    std::vector<float> queryOutputData(size, 0);
    ret = aclrtMemcpy(queryOutputData.data(), queryOutputData.size() * sizeof(queryOutputData[0]),
                      queryOutputDeviceAddr, size * sizeof(queryOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy queryOutput result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("queryOutput result[%ld] is: %f\n", i, queryOutputData[i]);
    }
    size = GetShapeSize(keyOutputShape);
    std::vector<float> keyOutputData(size, 0);
    ret = aclrtMemcpy(keyOutputData.data(), keyOutputData.size() * sizeof(keyOutputData[0]), keyOutputDeviceAddr,
                      size * sizeof(keyOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy keyOutput result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("keyOutput result[%ld] is: %f\n", i, keyOutputData[i]);
    }
    size = GetShapeSize(valueOutputShape);
    std::vector<float> valueOutputData(size, 0);
    ret = aclrtMemcpy(valueOutputData.data(), valueOutputData.size() * sizeof(valueOutputData[0]),
                      valueOutputDeviceAddr, size * sizeof(valueOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy valueOutput result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("valueOutput result[%ld] is: %f\n", i, valueOutputData[i]);
    }
    size = GetShapeSize(normQueryMeanShape);
    std::vector<float> normQueryMeanData(size, 0);
    ret = aclrtMemcpy(normQueryMeanData.data(), normQueryMeanData.size() * sizeof(normQueryMeanData[0]),
                      normQueryMeanDeviceAddr, size * sizeof(normQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryMean result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normQueryMean result[%ld] is: %f\n", i, normQueryMeanData[i]);
    }
    size = GetShapeSize(normQueryRstdShape);
    std::vector<float> normQueryRstdData(size, 0);
    ret = aclrtMemcpy(normQueryRstdData.data(), normQueryRstdData.size() * sizeof(normQueryRstdData[0]),
                      normQueryRstdDeviceAddr, size * sizeof(normQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryRstd result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normQueryRstd result[%ld] is: %f\n", i, normQueryRstdData[i]);
    }
    size = GetShapeSize(normKeyMeanShape);
    std::vector<float> normKeyMeanData(size, 0);
    ret = aclrtMemcpy(normKeyMeanData.data(), normKeyMeanData.size() * sizeof(normKeyMeanData[0]),
                      normKeyMeanDeviceAddr, size * sizeof(normKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyMean result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normKeyMean result[%ld] is: %f\n", i, normKeyMeanData[i]);
    }
    size = GetShapeSize(normKeyRstdShape);
    std::vector<float> normKeyRstdData(size, 0);
    ret = aclrtMemcpy(normKeyRstdData.data(), normKeyRstdData.size() * sizeof(normKeyRstdData[0]),
                      normKeyRstdDeviceAddr, size * sizeof(normKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyRstd result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normKeyRstd result[%ld] is: %f\n", i, normKeyRstdData[i]);
    }
    size = GetShapeSize(normAddedQueryMeanShape);
    std::vector<float> normAddedQueryMeanData(size, 0);
    ret =
        aclrtMemcpy(normAddedQueryMeanData.data(), normAddedQueryMeanData.size() * sizeof(normAddedQueryMeanData[0]),
                    normAddedQueryMeanDeviceAddr, size * sizeof(normAddedQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy normAddedQueryMean result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normAddedQueryMean result[%ld] is: %f\n", i, normAddedQueryMeanData[i]);
    }
    size = GetShapeSize(normAddedQueryRstdShape);
    std::vector<float> normAddedQueryRstdData(size, 0);
    ret =
        aclrtMemcpy(normAddedQueryRstdData.data(), normAddedQueryRstdData.size() * sizeof(normAddedQueryRstdData[0]),
                    normAddedQueryRstdDeviceAddr, size * sizeof(normAddedQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy normAddedQueryRstd result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normAddedQueryRstd result[%ld] is: %f\n", i, normAddedQueryRstdData[i]);
    }
    size = GetShapeSize(normAddedKeyMeanShape);
    std::vector<float> normAddedKeyMeanData(size, 0);
    ret = aclrtMemcpy(normAddedKeyMeanData.data(), normAddedKeyMeanData.size() * sizeof(normAddedKeyMeanData[0]),
                      normAddedKeyMeanDeviceAddr, size * sizeof(normAddedKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy normAddedKeyMean result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normAddedKeyMean result[%ld] is: %f\n", i, normAddedKeyMeanData[i]);
    }
    size = GetShapeSize(normAddedKeyRstdShape);
    std::vector<float> normAddedKeyRstdData(size, 0);
    ret = aclrtMemcpy(normAddedKeyRstdData.data(), normAddedKeyRstdData.size() * sizeof(normAddedKeyRstdData[0]),
                      normAddedKeyRstdDeviceAddr, size * sizeof(normAddedKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy normAddedKeyRstd result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("normAddedKeyRstd result[%ld] is: %f\n", i, normAddedKeyRstdData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(encoderQuery);
    aclDestroyTensor(encoderKey);
    aclDestroyTensor(encoderValue);
    aclDestroyTensor(normQueryWeight);
    aclDestroyTensor(normQueryBias);
    aclDestroyTensor(normKeyWeight);
    aclDestroyTensor(normKeyBias);
    aclDestroyTensor(normAddedQueryWeight);
    aclDestroyTensor(normAddedQueryBias);
    aclDestroyTensor(normAddedKeyWeight);
    aclDestroyTensor(normAddedKeyBias);
    aclDestroyTensor(ropeSin);
    aclDestroyTensor(ropeCos);
    aclDestroyTensor(queryOutput);
    aclDestroyTensor(keyOutput);
    aclDestroyTensor(valueOutput);
    aclDestroyTensor(normQueryMean);
    aclDestroyTensor(normQueryRstd);
    aclDestroyTensor(normKeyMean);
    aclDestroyTensor(normKeyRstd);
    aclDestroyTensor(normAddedQueryMean);
    aclDestroyTensor(normAddedQueryRstd);
    aclDestroyTensor(normAddedKeyMean);
    aclDestroyTensor(normAddedKeyRstd);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(encoderQueryDeviceAddr);
    aclrtFree(encoderKeyDeviceAddr);
    aclrtFree(encoderValueDeviceAddr);
    aclrtFree(normQueryWeightDeviceAddr);
    aclrtFree(normQueryBiasDeviceAddr);
    aclrtFree(normKeyWeightDeviceAddr);
    aclrtFree(normKeyBiasDeviceAddr);
    aclrtFree(normAddedQueryWeightDeviceAddr);
    aclrtFree(normAddedQueryBiasDeviceAddr);
    aclrtFree(normAddedKeyWeightDeviceAddr);
    aclrtFree(normAddedKeyBiasDeviceAddr);
    aclrtFree(ropeSinDeviceAddr);
    aclrtFree(ropeCosDeviceAddr);
    aclrtFree(queryOutputDeviceAddr);
    aclrtFree(keyOutputDeviceAddr);
    aclrtFree(valueOutputDeviceAddr);
    aclrtFree(normQueryMeanDeviceAddr);
    aclrtFree(normQueryRstdDeviceAddr);
    aclrtFree(normKeyMeanDeviceAddr);
    aclrtFree(normKeyRstdDeviceAddr);
    aclrtFree(normAddedQueryMeanDeviceAddr);
    aclrtFree(normAddedQueryRstdDeviceAddr);
    aclrtFree(normAddedKeyMeanDeviceAddr);
    aclrtFree(normAddedKeyRstdDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
