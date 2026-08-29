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
 * \file test_aclnn_mla_preprocess.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_mla_preprocess.h"

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

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T> &hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape) {
        fileSize *= i;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return false;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char *>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "读取文件失败" << std::endl;
        return false;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T> &hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char *>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

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

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, int num)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
int CreateAclTensorND(const std::vector<T> &shape, void **deviceAddr, void **hostAddr, aclDataType dataType,
                      aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ND tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ND tensor host failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy  failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T> &shape, void **deviceAddr, void **hostAddr, aclDataType dataType,
                      aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc NZ tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc NZ tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy  failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize)
{
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = 16;
    int64_t w0 = 32U / typeSize;
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 5;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 属性
    int64_t tokenNum = 8;
    int64_t hiddenNum = 7168;
    int64_t headNum = 32;
    int64_t blockNum = 192;
    int64_t blockSize = 128;

    int64_t wdqDim = 1536;
    int64_t qRopeDim = 64;
    int64_t kRopeDim = 64;
    double epsilon = 1e-05;
    int64_t qRotaryCoeff = 2;
    int64_t kRotaryCoeff = 2;
    bool transposeWdq = true;
    bool transposeWuq = true;
    bool transposeWuk = true;
    int64_t cacheMode = 1;
    int64_t quantMode = 0;
    bool doRmsNorm = true;
    int64_t wdkvSplitCount = 1;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {tokenNum, hiddenNum};
    std::vector<int64_t> gamma0Shape = {hiddenNum};
    std::vector<int64_t> beta0Shape = {hiddenNum};
    std::vector<int64_t> quantScale0Shape = {1};
    std::vector<int64_t> quantOffset0Shape = {1};
    std::vector<int64_t> wdqkvShape = {2112, hiddenNum};
    std::vector<int64_t> deScale0Shape = {2112};
    std::vector<int64_t> bias0Shape = {2112};
    std::vector<int64_t> gamma1Shape = {1536};
    std::vector<int64_t> beta1Shape = {1536};
    std::vector<int64_t> quantScale1Shape = {1};
    std::vector<int64_t> quantOffset1Shape = {1};
    std::vector<int64_t> wuqShape = {headNum * 192, 1536};
    std::vector<int64_t> deScale1Shape = {headNum * 192};
    std::vector<int64_t> bias1Shape = {headNum * 192};
    std::vector<int64_t> gamma2Shape = {512};
    std::vector<int64_t> cosShape = {tokenNum, 64};
    std::vector<int64_t> sinShape = {tokenNum, 64};
    std::vector<int64_t> wukShape = {headNum, 128, 512};
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, 1, 576};
    std::vector<int64_t> kvCacheRopeShape = {blockNum, blockSize, 1, 64};
    std::vector<int64_t> slotmappingShape = {tokenNum};
    std::vector<int64_t> ctkvScaleShape = {1};
    std::vector<int64_t> qNopeScaleShape = {1};

    std::vector<int64_t> qOutShape = {tokenNum, headNum, 576};
    std::vector<int64_t> kvCacheOutShape = {blockNum, blockSize, 1, 576};
    std::vector<int64_t> qRopeOutShape = {tokenNum, headNum, 64};
    std::vector<int64_t> krCacheOutShape = {blockNum, blockSize, 1, 64};

    void *inputDeviceAddr = nullptr;
    void *gamma0DeviceAddr = nullptr;
    void *beta0DeviceAddr = nullptr;
    void *quantScale0DeviceAddr = nullptr;
    void *quantOffset0DeviceAddr = nullptr;
    void *wdqkvDeviceAddr = nullptr;
    void *deScale0DeviceAddr = nullptr;
    void *bias0DeviceAddr = nullptr;
    void *gamma1DeviceAddr = nullptr;
    void *beta1DeviceAddr = nullptr;
    void *quantScale1DeviceAddr = nullptr;
    void *quantOffset1DeviceAddr = nullptr;
    void *wuqDeviceAddr = nullptr;
    void *deScale1DeviceAddr = nullptr;
    void *bias1DeviceAddr = nullptr;
    void *gamma2DeviceAddr = nullptr;
    void *cosDeviceAddr = nullptr;
    void *sinDeviceAddr = nullptr;
    void *wukDeviceAddr = nullptr;
    void *kvCacheDeviceAddr = nullptr;
    void *kvCacheRopeDeviceAddr = nullptr;
    void *slotmappingDeviceAddr = nullptr;
    void *ctkvScaleDeviceAddr = nullptr;
    void *qNopeScaleDeviceAddr = nullptr;
    void *qOutDeviceAddr = nullptr;
    void *kvCacheOutDeviceAddr = nullptr;
    void *qRopeOutDeviceAddr = nullptr;
    void *krCacheOutDeviceAddr = nullptr;

    void *inputHostAddr = nullptr;
    void *gamma0HostAddr = nullptr;
    void *beta0HostAddr = nullptr;
    void *quantScale0HostAddr = nullptr;
    void *quantOffset0HostAddr = nullptr;
    void *wdqkvHostAddr = nullptr;
    void *deScale0HostAddr = nullptr;
    void *bias0HostAddr = nullptr;
    void *gamma1HostAddr = nullptr;
    void *beta1HostAddr = nullptr;
    void *quantScale1HostAddr = nullptr;
    void *quantOffset1HostAddr = nullptr;
    void *wuqHostAddr = nullptr;
    void *deScale1HostAddr = nullptr;
    void *bias1HostAddr = nullptr;
    void *gamma2HostAddr = nullptr;
    void *cosHostAddr = nullptr;
    void *sinHostAddr = nullptr;
    void *wukHostAddr = nullptr;
    void *kvCacheHostAddr = nullptr;
    void *kvCacheRopeHostAddr = nullptr;
    void *slotmappingHostAddr = nullptr;
    void *ctkvScaleHostAddr = nullptr;
    void *qNopeScaleHostAddr = nullptr;
    void *qOutHostAddr = nullptr;
    void *kvCacheOutHostAddr = nullptr;
    void *qRopeOutHostAddr = nullptr;
    void *krCacheOutHostAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *gamma0 = nullptr;
    aclTensor *beta0 = nullptr;
    aclTensor *quantScale0 = nullptr;
    aclTensor *quantOffset0 = nullptr;
    aclTensor *wdqkv = nullptr;
    aclTensor *deScale0 = nullptr;
    aclTensor *bias0 = nullptr;
    aclTensor *gamma1 = nullptr;
    aclTensor *beta1 = nullptr;
    aclTensor *quantScale1 = nullptr;
    aclTensor *quantOffset1 = nullptr;
    aclTensor *wuq = nullptr;
    aclTensor *deScale1 = nullptr;
    aclTensor *bias1 = nullptr;
    aclTensor *gamma2 = nullptr;
    aclTensor *cos = nullptr;
    aclTensor *sin = nullptr;
    aclTensor *wuk = nullptr;
    aclTensor *kvCache = nullptr;
    aclTensor *kvCacheRope = nullptr;
    aclTensor *slotmapping = nullptr;
    aclTensor *ctkvScale = nullptr;
    aclTensor *qNopeScale = nullptr;
    aclTensor *qOut = nullptr;
    aclTensor *kvCacheOut = nullptr;
    aclTensor *qRopeOut = nullptr;
    aclTensor *krCacheOut = nullptr;

    // 转换三个NZ格式变量的shape
    ret = TransToNZShape(wdqkvShape, sizeof(int8_t));
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed. \n"); return ret);
    ret = TransToNZShape(wuqShape, sizeof(int8_t));
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed. \n"); return ret);

    ret = CreateAclTensorND(inputShape, &inputDeviceAddr, &inputHostAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(gamma0Shape, &gamma0DeviceAddr, &gamma0HostAddr, aclDataType::ACL_FLOAT16, &gamma0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(beta0Shape, &beta0DeviceAddr, &beta0HostAddr, aclDataType::ACL_FLOAT16, &beta0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(quantScale0Shape, &quantScale0DeviceAddr, &quantScale0HostAddr, aclDataType::ACL_FLOAT16,
                            &quantScale0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(quantOffset0Shape, &quantOffset0DeviceAddr, &quantOffset0HostAddr, aclDataType::ACL_INT8,
                            &quantOffset0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // wdqkv转为NZ
    ret = CreateAclTensorNZ(wdqkvShape, &wdqkvDeviceAddr, &wdqkvHostAddr, aclDataType::ACL_INT8, &wdqkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // fp16输入，则这里转为int64
    ret = CreateAclTensorND(deScale0Shape, &deScale0DeviceAddr, &deScale0HostAddr, aclDataType::ACL_INT64, &deScale0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(bias0Shape, &bias0DeviceAddr, &bias0HostAddr, aclDataType::ACL_INT32, &bias0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(gamma1Shape, &gamma1DeviceAddr, &gamma1HostAddr, aclDataType::ACL_FLOAT16, &gamma1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(beta1Shape, &beta1DeviceAddr, &beta1HostAddr, aclDataType::ACL_FLOAT16, &beta1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(quantScale1Shape, &quantScale1DeviceAddr, &quantScale1HostAddr, aclDataType::ACL_FLOAT16,
                            &quantScale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(quantOffset1Shape, &quantOffset1DeviceAddr, &quantOffset1HostAddr, aclDataType::ACL_INT8,
                            &quantOffset1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // wuq转为NZ
    ret = CreateAclTensorNZ(wuqShape, &wuqDeviceAddr, &wuqHostAddr, aclDataType::ACL_INT8, &wuq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // fp16输入，则这里转为int64
    ret = CreateAclTensorND(deScale1Shape, &deScale1DeviceAddr, &deScale1HostAddr, aclDataType::ACL_INT64, &deScale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(bias1Shape, &bias1DeviceAddr, &bias1HostAddr, aclDataType::ACL_INT32, &bias1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(gamma2Shape, &gamma2DeviceAddr, &gamma2HostAddr, aclDataType::ACL_FLOAT16, &gamma2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(cosShape, &cosDeviceAddr, &cosHostAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(sinShape, &sinDeviceAddr, &sinHostAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(wukShape, &wukDeviceAddr, &wukHostAddr, aclDataType::ACL_FLOAT16, &wuk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_FLOAT16, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(kvCacheRopeShape, &kvCacheRopeDeviceAddr, &kvCacheRopeHostAddr, aclDataType::ACL_FLOAT16,
                            &kvCacheRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(slotmappingShape, &slotmappingDeviceAddr, &slotmappingHostAddr, aclDataType::ACL_INT32,
                            &slotmapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(ctkvScaleShape, &ctkvScaleDeviceAddr, &ctkvScaleHostAddr, aclDataType::ACL_FLOAT16,
                            &ctkvScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(qNopeScaleShape, &qNopeScaleDeviceAddr, &qNopeScaleHostAddr, aclDataType::ACL_FLOAT16,
                            &qNopeScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(qOutShape, &qOutDeviceAddr, &qOutHostAddr, aclDataType::ACL_FLOAT16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(kvCacheOutShape, &kvCacheOutDeviceAddr, &kvCacheOutHostAddr, aclDataType::ACL_FLOAT16,
                            &kvCacheOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(qRopeOutShape, &qRopeOutDeviceAddr, &qRopeOutHostAddr, aclDataType::ACL_FLOAT16, &qRopeOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorND(krCacheOutShape, &krCacheOutDeviceAddr, &krCacheOutHostAddr, aclDataType::ACL_FLOAT16,
                            &krCacheOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 调用aclnnMlaPreprocess第一段接口
    ret = aclnnMlaPreprocessGetWorkspaceSize(
        input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0, gamma1, beta1, quantScale1,
        quantOffset1, wuq, deScale1, bias1, gamma2, cos, sin, wuk, kvCache, kvCacheRope, slotmapping, ctkvScale,
        qNopeScale, wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff, transposeWdq, transposeWuq,
        transposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount, qOut, kvCacheOut, qRopeOut, krCacheOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMlaPreprocess第二段接口
    ret = aclnnMlaPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPreprocess failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto qOutSize = GetShapeSize(qOutShape);
    std::vector<float> qOutData(qOutSize, 0);
    ret = aclrtMemcpy(qOutData.data(), qOutData.size() * sizeof(qOutData[0]), qOutDeviceAddr, qOutSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    // 释放aclTensor资源
    aclDestroyTensor(input);
    aclDestroyTensor(gamma0);
    aclDestroyTensor(beta0);
    aclDestroyTensor(quantScale0);
    aclDestroyTensor(quantOffset0);
    aclDestroyTensor(wdqkv);
    aclDestroyTensor(deScale0);
    aclDestroyTensor(bias0);
    aclDestroyTensor(gamma1);
    aclDestroyTensor(beta1);
    aclDestroyTensor(quantScale1);
    aclDestroyTensor(quantOffset1);
    aclDestroyTensor(wuq);
    aclDestroyTensor(deScale1);
    aclDestroyTensor(bias1);
    aclDestroyTensor(gamma2);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(wuk);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(kvCacheRope);
    aclDestroyTensor(slotmapping);
    aclDestroyTensor(ctkvScale);
    aclDestroyTensor(qNopeScale);

    // 7. 释放device 资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(gamma0DeviceAddr);
    aclrtFree(beta0DeviceAddr);
    aclrtFree(quantScale0DeviceAddr);
    aclrtFree(quantOffset0DeviceAddr);
    aclrtFree(wdqkvDeviceAddr);
    aclrtFree(deScale0DeviceAddr);
    aclrtFree(bias0DeviceAddr);
    aclrtFree(gamma1DeviceAddr);
    aclrtFree(beta1DeviceAddr);
    aclrtFree(quantScale1DeviceAddr);
    aclrtFree(quantOffset1DeviceAddr);
    aclrtFree(wuqDeviceAddr);
    aclrtFree(deScale1DeviceAddr);
    aclrtFree(bias1DeviceAddr);
    aclrtFree(gamma2DeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(wukDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(kvCacheRopeDeviceAddr);
    aclrtFree(slotmappingDeviceAddr);
    aclrtFree(ctkvScaleDeviceAddr);
    aclrtFree(qNopeScaleDeviceAddr);

    // 8. 释放host 资源
    aclrtFree(inputHostAddr);
    aclrtFree(gamma0HostAddr);
    aclrtFree(beta0HostAddr);
    aclrtFree(quantScale0HostAddr);
    aclrtFree(quantOffset0HostAddr);
    aclrtFree(wdqkvHostAddr);
    aclrtFree(deScale0HostAddr);
    aclrtFree(bias0HostAddr);
    aclrtFree(gamma1HostAddr);
    aclrtFree(beta1HostAddr);
    aclrtFree(quantScale1HostAddr);
    aclrtFree(quantOffset1HostAddr);
    aclrtFree(wuqHostAddr);
    aclrtFree(deScale1HostAddr);
    aclrtFree(bias1HostAddr);
    aclrtFree(gamma2HostAddr);
    aclrtFree(cosHostAddr);
    aclrtFree(sinHostAddr);
    aclrtFree(wukHostAddr);
    aclrtFree(kvCacheHostAddr);
    aclrtFree(kvCacheRopeHostAddr);
    aclrtFree(slotmappingHostAddr);
    aclrtFree(ctkvScaleHostAddr);
    aclrtFree(qNopeScaleHostAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
