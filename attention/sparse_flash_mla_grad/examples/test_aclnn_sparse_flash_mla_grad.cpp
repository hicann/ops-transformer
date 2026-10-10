/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_sparse_flash_mla_grad.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

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
    std::vector<short> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
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
    // 1. （固定写法）device/context/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> qShape = {1, 16, 512};       // T1, N1, D
    std::vector<int64_t> oriKvShape = {2048, 1, 512}; // T2, N2, D
    std::vector<int64_t> cmpKvShape = {16, 1, 512};   // T3, N2, D
    std::vector<int64_t> dOutShape = {1, 16, 512};    // T1, N1, D
    std::vector<int64_t> outShape = {1, 16, 512};     // T1, N1, D
    std::vector<int64_t> lseShape = {1, 1, 16};       // N2, T1, G
    std::vector<int64_t> cuSeqQLenshape = {2};        // B + 1
    std::vector<int64_t> cuSeqOriKvLenshape = {2};    // B + 1
    std::vector<int64_t> cuSeqCmpKvLenshape = {2};    // B + 1
    std::vector<int64_t> cmpResidualKvShape = {1};    // B
    std::vector<int64_t> sinksShape = {16};           // N1
    std::vector<int64_t> metadataShape = {1024};      // 固定大小

    void* qDeviceAddr = nullptr;
    void* oriKvDeviceAddr = nullptr;
    void* cmpKvDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dOutDeviceAddr = nullptr;
    void* lseDeviceAddr = nullptr;
    void* cuSeqQLenDeviceAddr = nullptr;
    void* cuSeqOriKvLenDeviceAddr = nullptr;
    void* cuSeqCmpKvLenDeviceAddr = nullptr;
    void* cmpResidualKvDeviceAddr = nullptr;
    void* sinksDeviceAddr = nullptr;
    void* dqDeviceAddr = nullptr;
    void* dOriKvDeviceAddr = nullptr;
    void* dCmpKvDeviceAddr = nullptr;
    void* dSinksDeviceAddr = nullptr;
    void* metadataDeviceAddr = nullptr;
    void* oriSoftmaxL1NormDeviceAddr = nullptr;
    void* cmpSoftmaxL1NormDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* oriKv = nullptr;
    aclTensor* cmpKv = nullptr;
    aclTensor* out = nullptr;
    aclTensor* dOut = nullptr;
    aclTensor* lse = nullptr;
    aclTensor* cuSeqQLen = nullptr;
    aclTensor* cuSeqOriKvLen = nullptr;
    aclTensor* cuSeqCmpKvLen = nullptr;
    aclTensor* cmpResidualKv = nullptr;
    aclTensor* sinks = nullptr;
    aclTensor* dq = nullptr;
    aclTensor* dOriKv = nullptr;
    aclTensor* dCmpKv = nullptr;
    aclTensor* dSinks = nullptr;
    aclTensor* metadata = nullptr;
    aclTensor* oriSoftmaxL1Norm = nullptr;
    aclTensor* cmpSoftmaxL1Norm = nullptr;

    std::vector<short> qHostData(1 * 16 * 512, 1.0);
    std::vector<short> oriKvHostData(2048 * 1 * 512, 1.0);
    std::vector<short> cmpKvHostData(16 * 1 * 512, 1.0);
    std::vector<short> outHostData(1 * 16 * 512, 1.0);
    std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
    std::vector<float> lseHostData(16, 3.0);
    std::vector<int32_t> cuSeqQLenHostData = {0, 1};
    std::vector<int32_t> cuSeqOriKvLenHostData = {0, 2048};
    std::vector<int32_t> cuSeqCmpKvLenHostData = {0, 16};
    std::vector<int32_t> cmpResidualKvHostData = {0};
    std::vector<float> sinksHostData(16, 128.0);
    std::vector<short> dqHostData(1 * 16 * 512, 0);
    std::vector<short> dOriKvHostData(2048 * 1 * 512, 0);
    std::vector<short> dCmpKvHostData(16 * 1 * 512, 0);
    std::vector<float> dSinksHostData(16, 0);
    std::vector<int32_t> metadataHostData(1024, 0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT16, &oriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &cmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lseHostData, lseShape, &lseDeviceAddr, aclDataType::ACL_FLOAT, &lse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqOriKvLenHostData, cuSeqOriKvLenshape, &cuSeqOriKvLenDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqOriKvLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqCmpKvLenHostData, cuSeqCmpKvLenshape, &cuSeqCmpKvLenDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqCmpKvLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32,
                          &cmpResidualKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dOriKvHostData, oriKvShape, &dOriKvDeviceAddr, aclDataType::ACL_FLOAT16, &dOriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dCmpKvHostData, cmpKvShape, &dCmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &dCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dSinksHostData, sinksShape, &dSinksDeviceAddr, aclDataType::ACL_FLOAT, &dSinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double scaleValue = 0.088388;
    int64_t cmpRatio = 128;
    int64_t oriMaskMode = 4;
    int64_t cmpMaskMode = 3;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
    char layoutQ[5] = {'T', 'N', 'D', 0};
    char layoutKv[5] = {'T', 'N', 'D', 0};

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 3.1 调用SparseFlashMlaGradMetadata获取metadata
    uint64_t metaWorkspaceSize = 0;
    aclOpExecutor* metaExecutor;
    ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
        cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen, nullptr, nullptr, nullptr, cmpResidualKv, nullptr, nullptr, 16, 1, 512,
        0, 0, 0, 0, 0, 0, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ, layoutKv, true, true,
        metadata, &metaWorkspaceSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* metaWorkspaceAddr = nullptr;
    if (metaWorkspaceSize > 0) {
        ret = aclrtMalloc(&metaWorkspaceAddr, metaWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSparseFlashMlaGradMetadata(metaWorkspaceAddr, metaWorkspaceSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadata failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 3.2 调用SparseFlashMlaGrad计算梯度
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSparseFlashMlaGradGetWorkspaceSize(
        q, dOut, out, lse, oriKv, cmpKv, nullptr, nullptr, cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen, nullptr, nullptr,
        nullptr, cmpResidualKv, nullptr, nullptr, sinks, metadata, scaleValue, cmpRatio, oriMaskMode, cmpMaskMode,
        oriWinLeft, oriWinRight, layoutQ, layoutKv, dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSparseFlashMlaGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(qShape, &dqDeviceAddr);
    PrintOutResult(cmpKvShape, &dCmpKvDeviceAddr);
    PrintOutResult(sinksShape, &dSinksDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(oriKv);
    aclDestroyTensor(cmpKv);
    aclDestroyTensor(out);
    aclDestroyTensor(dOut);
    aclDestroyTensor(lse);
    aclDestroyTensor(cuSeqQLen);
    aclDestroyTensor(cuSeqOriKvLen);
    aclDestroyTensor(cuSeqCmpKvLen);
    aclDestroyTensor(cmpResidualKv);
    aclDestroyTensor(sinks);
    aclDestroyTensor(dq);
    aclDestroyTensor(dOriKv);
    aclDestroyTensor(dCmpKv);
    aclDestroyTensor(dSinks);
    aclDestroyTensor(metadata);
    aclDestroyTensor(oriSoftmaxL1Norm);
    aclDestroyTensor(cmpSoftmaxL1Norm);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(oriKvDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(dOutDeviceAddr);
    aclrtFree(lseDeviceAddr);
    aclrtFree(cuSeqQLenDeviceAddr);
    aclrtFree(cuSeqOriKvLenDeviceAddr);
    aclrtFree(cuSeqCmpKvLenDeviceAddr);
    aclrtFree(cmpResidualKvDeviceAddr);
    aclrtFree(sinksDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dOriKvDeviceAddr);
    aclrtFree(dCmpKvDeviceAddr);
    aclrtFree(dSinksDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(oriSoftmaxL1NormDeviceAddr);
    aclrtFree(cmpSoftmaxL1NormDeviceAddr);
    if (metaWorkspaceSize > 0) {
        aclrtFree(metaWorkspaceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
