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
#include "aclnnop/aclnn_swin_transformer_ln_qkv_quant.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 49, 32};
    std::vector<int64_t> gammaShape = {32};
    std::vector<int64_t> weightShape = {32 * 3, 32};
    std::vector<int64_t> biasShape = {3 * 32};
    std::vector<int64_t> outShape = {1, 1, 49, 32};
    void *xDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;
    void *dequantDeviceAddr = nullptr;

    void *outqDeviceAddr = nullptr;
    void *outkDeviceAddr = nullptr;
    void *outvDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *quantScale = nullptr;
    aclTensor *quantOffset = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *queryOutput = nullptr;
    aclTensor *keyOutput = nullptr;
    aclTensor *valueOutput = nullptr;

    std::vector<uint16_t> selfHostData(49 * 32, 0x1);
    std::vector<int32_t> biasHostData(3 * 32, 0x1);
    std::vector<uint16_t> gammaHostData(32, 0x1);
    std::vector<uint16_t> betaHostData(32, 0x1);
    std::vector<int8_t> weightHostData(3 * 32 * 32, 0x1);
    std::vector<uint16_t> scaleHostData(32, 0x1);
    std::vector<uint16_t> offsetHostData(32, 0x1);
    std::vector<uint64_t> dequantHostData(3 * 32, 0x1);

    std::vector<uint16_t> outqHostData(49 * 32, 0x0);
    std::vector<uint16_t> outkHostData(49 * 32, 0x0);
    std::vector<uint16_t> outvHostData(49 * 32, 0x0);

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(scaleHostData, gammaShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(offsetHostData, gammaShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT16, &quantOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dequantHostData, biasShape, &dequantDeviceAddr, aclDataType::ACL_UINT64, &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outqHostData, outShape, &outqDeviceAddr, aclDataType::ACL_FLOAT16, &queryOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outkHostData, outShape, &outkDeviceAddr, aclDataType::ACL_FLOAT16, &keyOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outvHostData, outShape, &outvDeviceAddr, aclDataType::ACL_FLOAT16, &valueOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    float epsilon = 0.0001;
    int64_t oriHeight = 7;
    int64_t oriWeight = 7;
    int64_t hWinSize = 7;
    int64_t wWinSize = 7;
    int64_t headNum = 1;
    int64_t seqLength = 32;
    bool weightTranspose = true;

    // 调用aclnnSwinTransformerLnQkvQuant第一段接口
    ret = aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(
        x, gamma, beta, weight, bias, quantScale, quantOffset, dequantScale, headNum, seqLength, epsilon, oriHeight,
        oriWeight, hWinSize, wWinSize, weightTranspose, queryOutput, keyOutput, valueOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnSwinTransformerLnQkvQuant第二段接口
    ret = aclnnSwinTransformerLnQkvQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outqDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(quantScale);
    aclDestroyTensor(quantOffset);
    aclDestroyTensor(dequantScale);
    aclDestroyTensor(queryOutput);
    aclDestroyTensor(keyOutput);
    aclDestroyTensor(valueOutput);
    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(dequantDeviceAddr);
    aclrtFree(outqDeviceAddr);
    aclrtFree(outkDeviceAddr);
    aclrtFree(outvDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG_PRINT("test aclnn_swin_transformer_ln_qkv_quant finish. ret:%d\n", ret);
    return 0;
}
