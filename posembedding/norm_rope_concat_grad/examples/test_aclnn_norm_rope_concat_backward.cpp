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
 * \file test_aclnn_norm_rope_concat_backward.cpp
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
#include "aclnnop/aclnn_norm_rope_concat_grad.h"

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
                    aclDataType dataType, aclTensor** xOrResult)
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
    uint32_t normType = 2;
    uint32_t normAddedType = 2;
    uint32_t ropeType = 1;
    uint32_t concatOrder = 0;

    std::vector<int64_t> gradQueryOutputShape = {batchSize, headNum, querySeq + encoderQuerySeq, headDim};
    std::vector<int64_t> gradKeyOutputShape = {batchSize, headNum, keySeq + encoderKeySeq, headDim};
    std::vector<int64_t> gradValueOutputShape = {batchSize, headNum, valueSeq + encoderValueSeq, headDim};
    std::vector<int64_t> queryShape = {batchSize, querySeq, headNum, headDim};
    std::vector<int64_t> keyShape = {batchSize, keySeq, headNum, headDim};
    std::vector<int64_t> encoderQueryShape = {batchSize, encoderQuerySeq, headNum, headDim};
    std::vector<int64_t> encoderKeyShape = {batchSize, encoderKeySeq, headNum, headDim};
    std::vector<int64_t> normQueryWeightShape = {headDim};
    std::vector<int64_t> normQueryMeanShape = {batchSize, querySeq, headNum, 1};
    std::vector<int64_t> normQueryRstdShape = {batchSize, querySeq, headNum, 1};
    std::vector<int64_t> normKeyWeightShape = {headDim};
    std::vector<int64_t> normKeyMeanShape = {batchSize, keySeq, headNum, 1};
    std::vector<int64_t> normKeyRstdShape = {batchSize, keySeq, headNum, 1};
    std::vector<int64_t> normAddedQueryWeightShape = {headDim};
    std::vector<int64_t> normAddedQueryMeanShape = {batchSize, encoderQuerySeq, headNum, 1};
    std::vector<int64_t> normAddedQueryRstdShape = {batchSize, encoderQuerySeq, headNum, 1};
    std::vector<int64_t> normAddedKeyWeightShape = {headDim};
    std::vector<int64_t> normAddedKeyMeanShape = {batchSize, encoderKeySeq, headNum, 1};
    std::vector<int64_t> normAddedKeyRstdShape = {batchSize, encoderKeySeq, headNum, 1};
    std::vector<int64_t> ropeSinShape = {ropeSeq, headDim};
    std::vector<int64_t> ropeCosShape = {ropeSeq, headDim};
    std::vector<int64_t> gradQueryShape = {batchSize, querySeq, headNum, headDim};
    std::vector<int64_t> gradKeyShape = {batchSize, keySeq, headNum, headDim};
    std::vector<int64_t> gradValueShape = {batchSize, valueSeq, headNum, headDim};
    std::vector<int64_t> gradEncoderQueryShape = {batchSize, encoderQuerySeq, headNum, headDim};
    std::vector<int64_t> gradEncoderKeyShape = {batchSize, encoderKeySeq, headNum, headDim};
    std::vector<int64_t> gradEncoderValueShape = {batchSize, encoderValueSeq, headNum, headDim};
    std::vector<int64_t> gradNormQueryWeightShape = {headDim};
    std::vector<int64_t> gradNormQueryBiasShape = {headDim};
    std::vector<int64_t> gradNormKeyWeightShape = {headDim};
    std::vector<int64_t> gradNormKeyBiasShape = {headDim};
    std::vector<int64_t> gradNormAddedQueryWeightShape = {headDim};
    std::vector<int64_t> gradNormAddedQueryBiasShape = {headDim};
    std::vector<int64_t> gradNormAddedKeyWeightShape = {headDim};
    std::vector<int64_t> gradNormAddedKeyBiasShape = {headDim};

    void* gradQueryOutputDeviceAddr = nullptr;
    void* gradKeyOutputDeviceAddr = nullptr;
    void* gradValueOutputDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* encoderQueryDeviceAddr = nullptr;
    void* encoderKeyDeviceAddr = nullptr;
    void* normQueryWeightDeviceAddr = nullptr;
    void* normQueryMeanDeviceAddr = nullptr;
    void* normQueryRstdDeviceAddr = nullptr;
    void* normKeyWeightDeviceAddr = nullptr;
    void* normKeyMeanDeviceAddr = nullptr;
    void* normKeyRstdDeviceAddr = nullptr;
    void* normAddedQueryWeightDeviceAddr = nullptr;
    void* normAddedQueryMeanDeviceAddr = nullptr;
    void* normAddedQueryRstdDeviceAddr = nullptr;
    void* normAddedKeyWeightDeviceAddr = nullptr;
    void* normAddedKeyMeanDeviceAddr = nullptr;
    void* normAddedKeyRstdDeviceAddr = nullptr;
    void* ropeSinDeviceAddr = nullptr;
    void* ropeCosDeviceAddr = nullptr;
    void* gradQueryDeviceAddr = nullptr;
    void* gradKeyDeviceAddr = nullptr;
    void* gradValueDeviceAddr = nullptr;
    void* gradEncoderQueryDeviceAddr = nullptr;
    void* gradEncoderKeyDeviceAddr = nullptr;
    void* gradEncoderValueDeviceAddr = nullptr;
    void* gradNormQueryWeightDeviceAddr = nullptr;
    void* gradNormQueryBiasDeviceAddr = nullptr;
    void* gradNormKeyWeightDeviceAddr = nullptr;
    void* gradNormKeyBiasDeviceAddr = nullptr;
    void* gradNormAddedQueryWeightDeviceAddr = nullptr;
    void* gradNormAddedQueryBiasDeviceAddr = nullptr;
    void* gradNormAddedKeyWeightDeviceAddr = nullptr;
    void* gradNormAddedKeyBiasDeviceAddr = nullptr;

    aclTensor* gradQueryOutput = nullptr;
    aclTensor* gradKeyOutput = nullptr;
    aclTensor* gradValueOutput = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* encoderQuery = nullptr;
    aclTensor* encoderKey = nullptr;
    aclTensor* normQueryWeight = nullptr;
    aclTensor* normQueryMean = nullptr;
    aclTensor* normQueryRstd = nullptr;
    aclTensor* normKeyWeight = nullptr;
    aclTensor* normKeyMean = nullptr;
    aclTensor* normKeyRstd = nullptr;
    aclTensor* normAddedQueryWeight = nullptr;
    aclTensor* normAddedQueryMean = nullptr;
    aclTensor* normAddedQueryRstd = nullptr;
    aclTensor* normAddedKeyWeight = nullptr;
    aclTensor* normAddedKeyMean = nullptr;
    aclTensor* normAddedKeyRstd = nullptr;
    aclTensor* ropeSin = nullptr;
    aclTensor* ropeCos = nullptr;
    aclTensor* gradQuery = nullptr;
    aclTensor* gradKey = nullptr;
    aclTensor* gradValue = nullptr;
    aclTensor* gradEncoderQuery = nullptr;
    aclTensor* gradEncoderKey = nullptr;
    aclTensor* gradEncoderValue = nullptr;
    aclTensor* gradNormQueryWeight = nullptr;
    aclTensor* gradNormQueryBias = nullptr;
    aclTensor* gradNormKeyWeight = nullptr;
    aclTensor* gradNormKeyBias = nullptr;
    aclTensor* gradNormAddedQueryWeight = nullptr;
    aclTensor* gradNormAddedQueryBias = nullptr;
    aclTensor* gradNormAddedKeyWeight = nullptr;
    aclTensor* gradNormAddedKeyBias = nullptr;

    std::vector<float> gradQueryOutputHostData(batchSize * headNum * (querySeq + encoderQuerySeq) * headDim, 1.0);
    std::vector<float> gradKeyOutputHostData(batchSize * headNum * (keySeq + encoderKeySeq) * headDim, 1.0);
    std::vector<float> gradValueOutputHostData(batchSize * headNum * (valueSeq + encoderValueSeq) * headDim, 1.0);
    std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 1.0);
    std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 1.0);
    std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 1.0);
    std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 1.0);
    std::vector<float> normQueryWeightHostData(headDim, 1.0);
    std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
    std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 1.0);
    std::vector<float> normKeyWeightHostData(headDim, 1.0);
    std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
    std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 1.0);
    std::vector<float> normAddedQueryWeightHostData(headDim, 1.0);
    std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
    std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
    std::vector<float> normAddedKeyWeightHostData(headDim, 1.0);
    std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
    std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 1.0);
    std::vector<float> ropeSinHostData(ropeSeq * headDim, 1.0);
    std::vector<float> ropeCosHostData(ropeSeq * headDim, 1.0);
    std::vector<float> gradQueryHostData(batchSize * headNum * querySeq * headDim, 0.0);
    std::vector<float> gradKeyHostData(batchSize * headNum * keySeq * headDim, 0.0);
    std::vector<float> gradValueHostData(batchSize * headNum * valueSeq * headDim, 0.0);
    std::vector<float> gradEncoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 0.0);
    std::vector<float> gradEncoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 0.0);
    std::vector<float> gradEncoderValueHostData(batchSize * headNum * encoderValueSeq * headDim, 0.0);
    std::vector<float> gradNormQueryWeightHostData(headDim, 0.0);
    std::vector<float> gradNormQueryBiasHostData(headDim, 0.0);
    std::vector<float> gradNormKeyWeightHostData(headDim, 0.0);
    std::vector<float> gradNormKeyBiasHostData(headDim, 0.0);
    std::vector<float> gradNormAddedQueryWeightHostData(headDim, 0.0);
    std::vector<float> gradNormAddedQueryBiasHostData(headDim, 0.0);
    std::vector<float> gradNormAddedKeyWeightHostData(headDim, 0.0);
    std::vector<float> gradNormAddedKeyBiasHostData(headDim, 0.0);

    ret = CreateAclTensor(gradQueryOutputHostData, gradQueryOutputShape, &gradQueryOutputDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradQueryOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradKeyOutputHostData, gradKeyOutputShape, &gradKeyOutputDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradKeyOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradValueOutputHostData, gradValueOutputShape, &gradValueOutputDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradValueOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &encoderQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT,
                          &encoderKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &normQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
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
    ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradQueryHostData, gradQueryShape, &gradQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradKeyHostData, gradKeyShape, &gradKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradEncoderQueryHostData, gradEncoderQueryShape, &gradEncoderQueryDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradEncoderQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradEncoderKeyHostData, gradEncoderKeyShape, &gradEncoderKeyDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradEncoderKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradEncoderValueHostData, gradEncoderValueShape, &gradEncoderValueDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradEncoderValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormQueryWeightHostData, gradNormQueryWeightShape, &gradNormQueryWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradNormQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormQueryBiasHostData, gradNormQueryBiasShape, &gradNormQueryBiasDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradNormQueryBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormKeyWeightHostData, gradNormKeyWeightShape, &gradNormKeyWeightDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradNormKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormKeyBiasHostData, gradNormKeyBiasShape, &gradNormKeyBiasDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradNormKeyBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormAddedQueryWeightHostData, gradNormAddedQueryWeightShape,
                          &gradNormAddedQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormAddedQueryBiasHostData, gradNormAddedQueryBiasShape,
                          &gradNormAddedQueryBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormAddedKeyWeightHostData, gradNormAddedKeyWeightShape,
                          &gradNormAddedKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedKeyWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormAddedKeyBiasHostData, gradNormAddedKeyBiasShape, &gradNormAddedKeyBiasDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradNormAddedKeyBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGeGluBackward第一段接口
    ret = aclnnNormRopeConcatBackwardGetWorkspaceSize(
        gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
        normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight, normAddedQueryMean,
        normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin, ropeCos, normType,
        normAddedType, ropeType, concatOrder, gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey,
        gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias,
        gradNormAddedQueryWeight, gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGeGluBackward第二段接口
    ret = aclnnNormRopeConcatBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradQueryShape);
    std::vector<float> gradQueryData(size, 0);
    ret = aclrtMemcpy(gradQueryData.data(), gradQueryData.size() * sizeof(gradQueryData[0]), gradQueryDeviceAddr,
                      size * sizeof(gradQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradQuery result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradQuery result[%ld] is: %f\n", i, gradQueryData[i]);
    }
    size = GetShapeSize(gradKeyShape);
    std::vector<float> gradKeyData(size, 0);
    ret = aclrtMemcpy(gradKeyData.data(), gradKeyData.size() * sizeof(gradKeyData[0]), gradKeyDeviceAddr,
                      size * sizeof(gradKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradKey result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradKey result[%ld] is: %f\n", i, gradKeyData[i]);
    }
    size = GetShapeSize(gradValueShape);
    std::vector<float> gradValueData(size, 0);
    ret = aclrtMemcpy(gradValueData.data(), gradValueData.size() * sizeof(gradValueData[0]), gradValueDeviceAddr,
                      size * sizeof(gradValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradValue result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradValue result[%ld] is: %f\n", i, gradValueData[i]);
    }
    size = GetShapeSize(gradEncoderQueryShape);
    std::vector<float> gradEncoderQueryData(size, 0);
    ret = aclrtMemcpy(gradEncoderQueryData.data(), gradEncoderQueryData.size() * sizeof(gradEncoderQueryData[0]),
                      gradEncoderQueryDeviceAddr, size * sizeof(gradEncoderQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradEncoderQuery result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderQuery result[%ld] is: %f\n", i, gradEncoderQueryData[i]);
    }
    size = GetShapeSize(gradEncoderKeyShape);
    std::vector<float> gradEncoderKeyData(size, 0);
    ret = aclrtMemcpy(gradEncoderKeyData.data(), gradEncoderKeyData.size() * sizeof(gradEncoderKeyData[0]),
                      gradEncoderKeyDeviceAddr, size * sizeof(gradEncoderKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderKey result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderKey result[%ld] is: %f\n", i, gradEncoderKeyData[i]);
    }
    size = GetShapeSize(gradEncoderValueShape);
    std::vector<float> gradEncoderValueData(size, 0);
    ret = aclrtMemcpy(gradEncoderValueData.data(), gradEncoderValueData.size() * sizeof(gradEncoderValueData[0]),
                      gradEncoderValueDeviceAddr, size * sizeof(gradEncoderValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradEncoderValue result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderValue result[%ld] is: %f\n", i, gradEncoderValueData[i]);
    }
    size = GetShapeSize(gradNormQueryWeightShape);
    std::vector<float> gradNormQueryWeightData(size, 0);
    ret = aclrtMemcpy(
        gradNormQueryWeightData.data(), gradNormQueryWeightData.size() * sizeof(gradNormQueryWeightData[0]),
        gradNormQueryWeightDeviceAddr, size * sizeof(gradNormQueryWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormQueryWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryWeight result[%ld] is: %f\n", i, gradNormQueryWeightData[i]);
    }
    size = GetShapeSize(gradNormQueryBiasShape);
    std::vector<float> gradNormQueryBiasData(size, 0);
    ret = aclrtMemcpy(gradNormQueryBiasData.data(), gradNormQueryBiasData.size() * sizeof(gradNormQueryBiasData[0]),
                      gradNormQueryBiasDeviceAddr, size * sizeof(gradNormQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormQueryBias result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryBias result[%ld] is: %f\n", i, gradNormQueryBiasData[i]);
    }
    size = GetShapeSize(gradNormKeyWeightShape);
    std::vector<float> gradNormKeyWeightData(size, 0);
    ret = aclrtMemcpy(gradNormKeyWeightData.data(), gradNormKeyWeightData.size() * sizeof(gradNormKeyWeightData[0]),
                      gradNormKeyWeightDeviceAddr, size * sizeof(gradNormKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormKeyWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyWeight result[%ld] is: %f\n", i, gradNormKeyWeightData[i]);
    }
    size = GetShapeSize(gradNormKeyBiasShape);
    std::vector<float> gradNormKeyBiasData(size, 0);
    ret = aclrtMemcpy(gradNormKeyBiasData.data(), gradNormKeyBiasData.size() * sizeof(gradNormKeyBiasData[0]),
                      gradNormKeyBiasDeviceAddr, size * sizeof(gradNormKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormKeyBias result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyBias result[%ld] is: %f\n", i, gradNormKeyBiasData[i]);
    }
    size = GetShapeSize(gradNormAddedQueryWeightShape);
    std::vector<float> gradNormAddedQueryWeightData(size, 0);
    ret = aclrtMemcpy(gradNormAddedQueryWeightData.data(),
                      gradNormAddedQueryWeightData.size() * sizeof(gradNormAddedQueryWeightData[0]),
                      gradNormAddedQueryWeightDeviceAddr, size * sizeof(gradNormAddedQueryWeightData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormAddedQueryWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryWeight result[%ld] is: %f\n", i, gradNormAddedQueryWeightData[i]);
    }
    size = GetShapeSize(gradNormAddedQueryBiasShape);
    std::vector<float> gradNormAddedQueryBiasData(size, 0);
    ret = aclrtMemcpy(
        gradNormAddedQueryBiasData.data(), gradNormAddedQueryBiasData.size() * sizeof(gradNormAddedQueryBiasData[0]),
        gradNormAddedQueryBiasDeviceAddr, size * sizeof(gradNormAddedQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormAddedQueryBias result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryBias result[%ld] is: %f\n", i, gradNormAddedQueryBiasData[i]);
    }
    size = GetShapeSize(gradNormAddedKeyWeightShape);
    std::vector<float> gradNormAddedKeyWeightData(size, 0);
    ret = aclrtMemcpy(
        gradNormAddedKeyWeightData.data(), gradNormAddedKeyWeightData.size() * sizeof(gradNormAddedKeyWeightData[0]),
        gradNormAddedKeyWeightDeviceAddr, size * sizeof(gradNormAddedKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormAddedKeyWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyWeight result[%ld] is: %f\n", i, gradNormAddedKeyWeightData[i]);
    }
    size = GetShapeSize(gradNormAddedKeyBiasShape);
    std::vector<float> gradNormAddedKeyBiasData(size, 0);
    ret = aclrtMemcpy(
        gradNormAddedKeyBiasData.data(), gradNormAddedKeyBiasData.size() * sizeof(gradNormAddedKeyBiasData[0]),
        gradNormAddedKeyBiasDeviceAddr, size * sizeof(gradNormAddedKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy gradNormAddedKeyBias result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyBias result[%ld] is: %f\n", i, gradNormAddedKeyBiasData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradQueryOutput);
    aclDestroyTensor(gradKeyOutput);
    aclDestroyTensor(gradValueOutput);
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(encoderQuery);
    aclDestroyTensor(encoderKey);
    aclDestroyTensor(normQueryWeight);
    aclDestroyTensor(normQueryMean);
    aclDestroyTensor(normQueryRstd);
    aclDestroyTensor(normKeyWeight);
    aclDestroyTensor(normKeyMean);
    aclDestroyTensor(normKeyRstd);
    aclDestroyTensor(normAddedQueryWeight);
    aclDestroyTensor(normAddedQueryMean);
    aclDestroyTensor(normAddedQueryRstd);
    aclDestroyTensor(normAddedKeyWeight);
    aclDestroyTensor(normAddedKeyMean);
    aclDestroyTensor(normAddedKeyRstd);
    aclDestroyTensor(ropeSin);
    aclDestroyTensor(ropeCos);
    aclDestroyTensor(gradQuery);
    aclDestroyTensor(gradKey);
    aclDestroyTensor(gradValue);
    aclDestroyTensor(gradEncoderQuery);
    aclDestroyTensor(gradEncoderKey);
    aclDestroyTensor(gradEncoderValue);
    aclDestroyTensor(gradNormQueryWeight);
    aclDestroyTensor(gradNormQueryBias);
    aclDestroyTensor(gradNormKeyWeight);
    aclDestroyTensor(gradNormKeyBias);
    aclDestroyTensor(gradNormAddedQueryWeight);
    aclDestroyTensor(gradNormAddedQueryBias);
    aclDestroyTensor(gradNormAddedKeyWeight);
    aclDestroyTensor(gradNormAddedKeyBias);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradQueryOutputDeviceAddr);
    aclrtFree(gradKeyOutputDeviceAddr);
    aclrtFree(gradValueOutputDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(encoderQueryDeviceAddr);
    aclrtFree(encoderKeyDeviceAddr);
    aclrtFree(normQueryWeightDeviceAddr);
    aclrtFree(normQueryMeanDeviceAddr);
    aclrtFree(normQueryRstdDeviceAddr);
    aclrtFree(normKeyWeightDeviceAddr);
    aclrtFree(normKeyMeanDeviceAddr);
    aclrtFree(normKeyRstdDeviceAddr);
    aclrtFree(normAddedQueryWeightDeviceAddr);
    aclrtFree(normAddedQueryMeanDeviceAddr);
    aclrtFree(normAddedQueryRstdDeviceAddr);
    aclrtFree(normAddedKeyWeightDeviceAddr);
    aclrtFree(normAddedKeyMeanDeviceAddr);
    aclrtFree(normAddedKeyRstdDeviceAddr);
    aclrtFree(ropeSinDeviceAddr);
    aclrtFree(ropeCosDeviceAddr);
    aclrtFree(gradQueryDeviceAddr);
    aclrtFree(gradKeyDeviceAddr);
    aclrtFree(gradValueDeviceAddr);
    aclrtFree(gradEncoderQueryDeviceAddr);
    aclrtFree(gradEncoderKeyDeviceAddr);
    aclrtFree(gradEncoderValueDeviceAddr);
    aclrtFree(gradNormQueryWeightDeviceAddr);
    aclrtFree(gradNormQueryBiasDeviceAddr);
    aclrtFree(gradNormKeyWeightDeviceAddr);
    aclrtFree(gradNormKeyBiasDeviceAddr);
    aclrtFree(gradNormAddedQueryWeightDeviceAddr);
    aclrtFree(gradNormAddedQueryBiasDeviceAddr);
    aclrtFree(gradNormAddedKeyWeightDeviceAddr);
    aclrtFree(gradNormAddedKeyBiasDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
