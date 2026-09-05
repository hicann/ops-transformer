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
#include <algorithm>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals_grad.h"

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
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // SPLIT_H 用例：Ascend 910B 上 FP16、K=3 时，H=8192 超过 FULL_H 的 UB 容量。
    // Kernel 日志应输出：hMode=1, kernel=SPLIT_H。
    const int64_t B = 2;
    const int64_t N = 2;
    const int64_t H = 8192;
    const int64_t N1 = N + 1;
    const int64_t validBlockNum = N; // Reserved attribute; currently not used by the kernel.

    std::vector<int64_t> partialBlockShape = {B, H};
    std::vector<int64_t> blockResShape = {B, N, H};
    std::vector<int64_t> projWeightShape = {1, H};
    std::vector<int64_t> normWeightShape = {H};
    std::vector<int64_t> gradHiddenStatesShape = {B, H};
    std::vector<int64_t> invNormShape = {B, N1};
    std::vector<int64_t> probsShape = {B, N1};

    std::vector<uint16_t> partialBlockData(GetShapeSize(partialBlockShape), 0x3C00); // FP16 1.0
    std::vector<uint16_t> blockResData(GetShapeSize(blockResShape), 0x0000);
    std::vector<uint16_t> projWeightData(GetShapeSize(projWeightShape), 0x3C00);
    std::vector<uint16_t> normWeightData(GetShapeSize(normWeightShape), 0x3C00);
    std::vector<uint16_t> gradHiddenStatesData(GetShapeSize(gradHiddenStatesShape), 0x3C00);
    std::vector<float> invNormData(GetShapeSize(invNormShape), 1.0f);
    std::vector<float> probsData(GetShapeSize(probsShape), 0.0f);

    // 让三个 V block 的值不同，配合非均匀 probs 产生非零 gradScore 和 varianceScale。
    for (int64_t batch = 0; batch < B; ++batch) {
        std::fill_n(blockResData.begin() + (batch * N) * H, H, static_cast<uint16_t>(0x3800)); // FP16 0.5
        std::fill_n(blockResData.begin() + (batch * N + 1) * H, H,
                    static_cast<uint16_t>(0x3E00)); // FP16 1.5
        probsData[batch * N1] = 0.2f;
        probsData[batch * N1 + 1] = 0.3f;
        probsData[batch * N1 + 2] = 0.5f;
    }

    aclTensor *partial_block = nullptr, *block_res = nullptr, *proj_weight = nullptr, *norm_weight = nullptr;
    aclTensor *grad_hidden_states = nullptr, *inv_norm = nullptr, *probs = nullptr;
    void *d_partial_block = nullptr, *d_block_res = nullptr, *d_proj_weight = nullptr, *d_norm_weight = nullptr;
    void *d_grad_hidden_states = nullptr, *d_inv_norm = nullptr, *d_probs = nullptr;

    ret = CreateAclTensor(partialBlockData, partialBlockShape, &d_partial_block, aclDataType::ACL_FLOAT16,
                          &partial_block);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockResData, blockResShape, &d_block_res, aclDataType::ACL_FLOAT16, &block_res);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projWeightData, projWeightShape, &d_proj_weight, aclDataType::ACL_FLOAT16, &proj_weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, &d_norm_weight, aclDataType::ACL_FLOAT16, &norm_weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHiddenStatesData, gradHiddenStatesShape, &d_grad_hidden_states, aclDataType::ACL_FLOAT16,
                          &grad_hidden_states);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invNormData, invNormShape, &d_inv_norm, aclDataType::ACL_FLOAT, &inv_norm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsData, probsShape, &d_probs, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *grad_partial_block = nullptr, *grad_block_res = nullptr, *grad_proj_weight = nullptr,
              *grad_norm_weight = nullptr;
    void *d_grad_partial_block = nullptr, *d_grad_block_res = nullptr, *d_grad_proj_weight = nullptr,
         *d_grad_norm_weight = nullptr;

    std::vector<uint16_t> gradPartialBlockData(GetShapeSize(partialBlockShape), 0);
    std::vector<uint16_t> gradBlockResData(GetShapeSize(blockResShape), 0);
    std::vector<uint16_t> gradProjWeightData(GetShapeSize(projWeightShape), 0);
    std::vector<uint16_t> gradNormWeightData(GetShapeSize(normWeightShape), 0);

    ret = CreateAclTensor(gradPartialBlockData, partialBlockShape, &d_grad_partial_block, aclDataType::ACL_FLOAT16,
                          &grad_partial_block);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(gradBlockResData, blockResShape, &d_grad_block_res, aclDataType::ACL_FLOAT16, &grad_block_res);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradProjWeightData, projWeightShape, &d_grad_proj_weight, aclDataType::ACL_FLOAT16,
                          &grad_proj_weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormWeightData, normWeightShape, &d_grad_norm_weight, aclDataType::ACL_FLOAT16,
                          &grad_norm_weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    ret = aclnnBlockAttentionResidualsGradGetWorkspaceSize(
        partial_block, block_res, proj_weight, norm_weight, grad_hidden_states, inv_norm, probs, validBlockNum,
        grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttentionResidualsGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("Run SPLIT_H example: B=%ld, N=%ld, H=%ld, workspaceSize=%lu bytes\n", B, N, H, workspaceSize);

    void *workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBlockAttentionResidualsGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResidualsGrad SPLIT_H example ran successfully!\n");

    // clean up inputs
    aclDestroyTensor(partial_block);
    aclDestroyTensor(block_res);
    aclDestroyTensor(proj_weight);
    aclDestroyTensor(norm_weight);
    aclDestroyTensor(grad_hidden_states);
    aclDestroyTensor(inv_norm);
    aclDestroyTensor(probs);
    aclrtFree(d_partial_block);
    aclrtFree(d_block_res);
    aclrtFree(d_proj_weight);
    aclrtFree(d_norm_weight);
    aclrtFree(d_grad_hidden_states);
    aclrtFree(d_inv_norm);
    aclrtFree(d_probs);

    // clean up outputs
    aclDestroyTensor(grad_partial_block);
    aclDestroyTensor(grad_block_res);
    aclDestroyTensor(grad_proj_weight);
    aclDestroyTensor(grad_norm_weight);
    aclrtFree(d_grad_partial_block);
    aclrtFree(d_grad_block_res);
    aclrtFree(d_grad_proj_weight);
    aclrtFree(d_grad_norm_weight);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
