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
 * \file test_aclnn_block_sparse_attention_grad.cpp
 * \brief BlockSparseAttentionGrad 算子测试用例 (BNSD Layout)
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_grad.h"

using namespace std;

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
    // 固定写法，AscendCL初始化
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
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", hostData.size(),
                  GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return -1);
    return 0;
}

int main()
{
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置核心参数 (以 BNSD Layout 为例)
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t headDim = 128;
    int32_t blockShapeX = 64;
    int32_t blockShapeY = 64;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. 构建张量 Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // LSE 通常没有尾部 1 维，防止 GE squeeze
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. 分配并初始化 Host 数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // 将 Q, K, V 初始化为 0.1f 等较小的数
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);

    // 梯度初始值可以给一个小正数
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);

    // LSE 给一个合理的正数，比如 5.0f，这样 exp(S - LSE) 就是一个非常安全的负指数，绝对不会溢出
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // 创建所有的前向输入/输出 aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;

    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);

    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor);     // 严格使用 FP32
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // 严格使用 UINT8

    // 5. 创建反向输出梯度 (dq, dk, dv)
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);

    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. 创建 aclIntArray 属性参数 (BlockShape & ActualSeqLengths)
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    std::vector<int64_t> qSeqLenVec(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> kvSeqLenVec(batch, static_cast<int64_t>(kvSeqlen));
    aclIntArray *qSeqLenArr = aclCreateIntArray(qSeqLenVec.data(), batch);
    aclIntArray *kvSeqLenArr = aclCreateIntArray(kvSeqLenVec.data(), batch);

    // 7. 标量与字符串参数配置
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // 强制规定滑动窗口极大值
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;

    // 8. 调用第一段接口: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor, qTensor, kTensor, vTensor, outTensor, lseTensor,
        maskTensor, // blockSparseMaskOptional
        nullptr,    // attenMaskOptional 必须为空
        blockShapeArr, qSeqLenArr, kvSeqLenArr, qLayoutBuffer, kvLayoutBuffer, static_cast<int64_t>(numKvHeads),
        maskType, scaleValue, preTokens, nextTokens, dqTensor, dkTensor, dvTensor, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. 分配 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. 调用第二段接口: 执行计算
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. 同步 Stream，等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. 将结果拷贝回 Host 侧打印
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. 释放所有资源
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr)
        aclrtFree(workspaceAddr);

    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(vAddr);
    aclrtFree(doutAddr);
    aclrtFree(outAddr);
    aclrtFree(lseAddr);
    aclrtFree(maskAddr);
    aclrtFree(dqAddr);
    aclrtFree(dkAddr);
    aclrtFree(dvAddr);

    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor);
    aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor);
    aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor);
    aclDestroyTensor(dkTensor);
    aclDestroyTensor(dvTensor);

    aclDestroyIntArray(blockShapeArr);
    aclDestroyIntArray(qSeqLenArr);
    aclDestroyIntArray(kvSeqLenArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
