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
 * \file test_aclnn_block_attention_residuals.cpp
 * \brief aclnnBlockAttentionResiduals 调用示例。shape 维数与文档「维度(shape)」列一致：
 *        partialBlock/hiddenStates 为 2 维 (T,H)，blockRes 为 3 维 (T,N,H)，
 *        projWeight 为 2 维 (1,H)，normWeight 为 1 维 (H)。
 */
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals.h"

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
constexpr int64_t PARTIAL_BLOCK_DIM_NUM = 2;
constexpr int64_t BLOCK_RES_DIM_NUM = 3;
constexpr int64_t PROJ_WEIGHT_DIM_NUM = 2;
constexpr int64_t NORM_WEIGHT_DIM_NUM = 1;
constexpr int64_t HIDDEN_STATES_DIM_NUM = 2;
constexpr double DEFAULT_NORM_EPS = 1.0e-6;
constexpr bool NEED_BACKWARD = false;

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

    // 仓上示例通用做法：按维度从后向前累乘，生成 ND 连续 Tensor 的 strides。
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

int CheckDocDims(const std::vector<int64_t> &partialBlockShape, const std::vector<int64_t> &blockResShape,
                 const std::vector<int64_t> &projWeightShape, const std::vector<int64_t> &normWeightShape,
                 const std::vector<int64_t> &hiddenStatesShape, int64_t validBlockNum)
{
    CHECK_RET(static_cast<int64_t>(partialBlockShape.size()) == PARTIAL_BLOCK_DIM_NUM,
              LOG_PRINT("partialBlock dim num mismatch, expect %ld got %zu\n", PARTIAL_BLOCK_DIM_NUM,
                        partialBlockShape.size());
              return ACL_ERROR_INVALID_PARAM);
    CHECK_RET(static_cast<int64_t>(blockResShape.size()) == BLOCK_RES_DIM_NUM,
              LOG_PRINT("blockRes dim num mismatch, expect %ld got %zu\n", BLOCK_RES_DIM_NUM, blockResShape.size());
              return ACL_ERROR_INVALID_PARAM);
    CHECK_RET(
        static_cast<int64_t>(projWeightShape.size()) == PROJ_WEIGHT_DIM_NUM,
        LOG_PRINT("projWeight dim num mismatch, expect %ld got %zu\n", PROJ_WEIGHT_DIM_NUM, projWeightShape.size());
        return ACL_ERROR_INVALID_PARAM);
    CHECK_RET(
        static_cast<int64_t>(normWeightShape.size()) == NORM_WEIGHT_DIM_NUM,
        LOG_PRINT("normWeight dim num mismatch, expect %ld got %zu\n", NORM_WEIGHT_DIM_NUM, normWeightShape.size());
        return ACL_ERROR_INVALID_PARAM);
    CHECK_RET(static_cast<int64_t>(hiddenStatesShape.size()) == HIDDEN_STATES_DIM_NUM,
              LOG_PRINT("hiddenStates dim num mismatch, expect %ld got %zu\n", HIDDEN_STATES_DIM_NUM,
                        hiddenStatesShape.size());
              return ACL_ERROR_INVALID_PARAM);
    const int64_t t = partialBlockShape[0];
    const int64_t h = partialBlockShape[1];
    const int64_t n = blockResShape[1];
    CHECK_RET(blockResShape[0] == t && blockResShape[2] == h && projWeightShape[0] == 1 && projWeightShape[1] == h &&
                  normWeightShape[0] == h && hiddenStatesShape[0] == t && hiddenStatesShape[1] == h &&
                  (validBlockNum == -1 || validBlockNum == n),
              LOG_PRINT("example shapes are inconsistent with (T,N,H)\n");
              return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

int aclnnBlockAttentionResidualsTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 文档规格：T>=0、H>=1、1<=N<=100；本示例 T=2、N=4、H=64。
    constexpr int64_t T = 2;
    constexpr int64_t N = 4;
    constexpr int64_t H = 64;
    const std::vector<int64_t> partialBlockShape = {T, H};
    const std::vector<int64_t> blockResShape = {T, N, H};
    const std::vector<int64_t> projWeightShape = {1, H};
    const std::vector<int64_t> normWeightShape = {H};
    const std::vector<int64_t> hiddenStatesShape = {T, H};
    constexpr int64_t VALID_BLOCK_NUM = -1; // 默认-1，表示使用 blockRes 的 N
    ret = CheckDocDims(partialBlockShape, blockResShape, projWeightShape, normWeightShape, hiddenStatesShape,
                       VALID_BLOCK_NUM);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<uint16_t> partialBlockHostData(GetShapeSize(partialBlockShape), 0x3C00);
    std::vector<uint16_t> blockResHostData(GetShapeSize(blockResShape), 0x3C00);
    std::vector<uint16_t> projWeightHostData(GetShapeSize(projWeightShape), 0x3C00);
    std::vector<uint16_t> normWeightHostData(GetShapeSize(normWeightShape), 0x3C00);
    std::vector<uint16_t> hiddenStatesHostData(GetShapeSize(hiddenStatesShape), 0);

    void *partialBlockDeviceAddr = nullptr;
    void *blockResDeviceAddr = nullptr;
    void *projWeightDeviceAddr = nullptr;
    void *normWeightDeviceAddr = nullptr;
    void *hiddenStatesDeviceAddr = nullptr;
    aclTensor *partialBlock = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *projWeight = nullptr;
    aclTensor *normWeight = nullptr;
    aclTensor *hiddenStates = nullptr;

    ret = CreateAclTensor(partialBlockHostData, partialBlockShape, &partialBlockDeviceAddr, aclDataType::ACL_BF16,
                          &partialBlock);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> partialBlockTensorPtr(partialBlock,
                                                                                         aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> partialBlockDeviceAddrPtr(partialBlockDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(blockResHostData, blockResShape, &blockResDeviceAddr, aclDataType::ACL_BF16, &blockRes);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> blockResTensorPtr(blockRes, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> blockResDeviceAddrPtr(blockResDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(projWeightHostData, projWeightShape, &projWeightDeviceAddr, aclDataType::ACL_BF16, &projWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> projWeightTensorPtr(projWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> projWeightDeviceAddrPtr(projWeightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(normWeightHostData, normWeightShape, &normWeightDeviceAddr, aclDataType::ACL_BF16, &normWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> normWeightTensorPtr(normWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> normWeightDeviceAddrPtr(normWeightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hiddenStatesHostData, hiddenStatesShape, &hiddenStatesDeviceAddr, aclDataType::ACL_BF16,
                          &hiddenStates);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> hiddenStatesTensorPtr(hiddenStates,
                                                                                         aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> hiddenStatesDeviceAddrPtr(hiddenStatesDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttentionResidualsGetWorkspaceSize(partialBlock, blockRes, projWeight, normWeight, VALID_BLOCK_NUM,
                                                       DEFAULT_NORM_EPS, NEED_BACKWARD, hiddenStates, nullptr, nullptr,
                                                       &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttentionResiduals(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResiduals failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(hiddenStatesHostData.data(), hiddenStatesHostData.size() * sizeof(hiddenStatesHostData[0]),
                      hiddenStatesDeviceAddr, hiddenStatesHostData.size() * sizeof(hiddenStatesHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy hiddenStates from device to host failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("BlockAttentionResiduals example ran successfully, hidden_states elems=%ld\n",
              GetShapeSize(hiddenStatesShape));
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    const auto ret = aclnnBlockAttentionResidualsTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsTest failed. ERROR: %d\n", ret);
                   return ret);
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
