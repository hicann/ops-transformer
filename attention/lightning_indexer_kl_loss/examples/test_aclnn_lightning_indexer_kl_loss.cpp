/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_kl_loss.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
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
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/context/stream 初始化
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // target_score: (B, H, D) = (2, 4, 128), float32, 非归一化原始注意力分数
    // index_probs:  (B, H, D) = (2, 4, 128), float32, softmax 后的概率分布
    // loss:         标量 (1,), float32
    int64_t b = 2;
    int64_t h = 4;
    int64_t d = 128;
    std::vector<int64_t> inputShape = {b, h, d};
    std::vector<int64_t> lossShape = {1};

    // 构造非归一化的 target_score 数据（sum != 1）
    std::vector<float> targetScoreHostData(b * h * d, 0);
    for (int64_t i = 0; i < b * h * d; i++) {
        targetScoreHostData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f;
    }

    // 构造 index_probs: 用随机数据模拟 softmax 后的概率分布（和为 1）
    std::vector<float> indexProbsHostData(b * h * d, 0);
    for (int64_t bh = 0; bh < b * h; bh++) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            indexProbsHostData[bh * d + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            sum += indexProbsHostData[bh * d + j];
        }
        for (int64_t j = 0; j < d; j++) {
            indexProbsHostData[bh * d + j] /= sum;
        }
    }

    // loss 初始值任意
    std::vector<float> lossHostData(1, 0.0f);

    void *targetScoreDeviceAddr = nullptr;
    void *indexProbsDeviceAddr = nullptr;
    void *lossDeviceAddr = nullptr;

    aclTensor *targetScore = nullptr;
    aclTensor *indexProbs = nullptr;
    aclTensor *loss = nullptr;

    ret =
        CreateAclTensor(targetScoreHostData, inputShape, &targetScoreDeviceAddr, aclDataType::ACL_FLOAT, &targetScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indexProbsHostData, inputShape, &indexProbsDeviceAddr, aclDataType::ACL_FLOAT, &indexProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // eps 参数
    double eps = 1e-9;

    // weight_type 参数（默认 'logits'）
    const char *weightType = "logits";

    // 3. 调用 C(A)NN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 4. 调用 aclnnLightningIndexerKLLoss 第一段接口
    ret = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore, indexProbs, eps, weightType, loss, &workspaceSize,
                                                      &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据 workspaceSize 申请 device 内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnLightningIndexerKLLoss 第二段接口
    ret = aclnnLightningIndexerKLLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerKLLoss failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 获取输出结果
    PrintOutResult(lossShape, &lossDeviceAddr);

    // 8. 释放 aclTensor
    aclDestroyTensor(targetScore);
    aclDestroyTensor(indexProbs);
    aclDestroyTensor(loss);

    // 9. 释放 device 资源
    aclrtFree(targetScoreDeviceAddr);
    aclrtFree(indexProbsDeviceAddr);
    aclrtFree(lossDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
