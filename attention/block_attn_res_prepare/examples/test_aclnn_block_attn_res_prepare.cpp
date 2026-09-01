/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t ROW_MAJOR_STRIDE_START_OFFSET = 2;
constexpr float DEFAULT_EPS = 1.0e-6F;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - ROW_MAJOR_STRIDE_START_OFFSET; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnBlockAttnResPrepareTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入与输出：1个token、2个历史来源、2个目标slot，hidden size为4。
    const std::vector<int64_t> blockResShape = {1, 2, 4};
    const std::vector<int64_t> validBlocksShape = {1};
    const std::vector<int64_t> pseudoQueryShape = {2, 4};
    const std::vector<int64_t> numeratorShape = {2, 1, 4};
    const std::vector<int64_t> statsShape = {2, 1};

    const std::vector<float> blockResHostData = {
        1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 0.0F, -1.0F, 1.0F,
    };
    const std::vector<uint64_t> validBlocksHostData = {2U};
    const std::vector<float> pseudoQueryHostData = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
    };
    std::vector<float> numeratorHostData(GetShapeSize(numeratorShape), 0.0F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 0.0F);

    void *blockResDeviceAddr = nullptr;
    void *validBlocksDeviceAddr = nullptr;
    void *pseudoQueryDeviceAddr = nullptr;
    void *numeratorDeviceAddr = nullptr;
    void *logitMaxDeviceAddr = nullptr;
    void *expSumDeviceAddr = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *validBlocks = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;

    ret = CreateAclTensor(blockResHostData, blockResShape, &blockResDeviceAddr, aclDataType::ACL_FLOAT, &blockRes);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> blockResTensorPtr(blockRes, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> blockResDeviceAddrPtr(blockResDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(validBlocksHostData, validBlocksShape, &validBlocksDeviceAddr, aclDataType::ACL_UINT64,
                          &validBlocks);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> validBlocksTensorPtr(validBlocks, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> validBlocksDeviceAddrPtr(validBlocksDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(pseudoQueryHostData, pseudoQueryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &pseudoQuery);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(numeratorHostData, numeratorShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT, &numerator);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> numeratorTensorPtr(numerator, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitMax);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT, &expSum);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> expSumTensorPtr(expSum, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                                                   DEFAULT_EPS, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResPrepareGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttnResPrepare(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResPrepare failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(numeratorHostData.data(), numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      numeratorDeviceAddr, numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy numerator from device to host failed. ERROR: %d\n", ret); return ret);
    ret =
        aclrtMemcpy(logitMaxHostData.data(), logitMaxHostData.size() * sizeof(logitMaxHostData[0]), logitMaxDeviceAddr,
                    logitMaxHostData.size() * sizeof(logitMaxHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logitMax from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(expSumHostData.data(), expSumHostData.size() * sizeof(expSumHostData[0]), expSumDeviceAddr,
                      expSumHostData.size() * sizeof(expSumHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expSum from device to host failed. ERROR: %d\n", ret); return ret);

    for (size_t i = 0; i < numeratorHostData.size(); ++i) {
        LOG_PRINT("numerator[%zu] = %.6f\n", i, numeratorHostData[i]);
    }
    for (size_t i = 0; i < logitMaxHostData.size(); ++i) {
        LOG_PRINT("logit_max[%zu] = %.6f, exp_sum[%zu] = %.6f\n", i, logitMaxHostData[i], i, expSumHostData[i]);
    }
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    // 根据实际运行环境设置Device ID。
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    const auto ret = aclnnBlockAttnResPrepareTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResPrepareTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
