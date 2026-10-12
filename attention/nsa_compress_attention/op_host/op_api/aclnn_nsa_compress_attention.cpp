/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_nsa_compress_attention.h"
#include "nsa_compress_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DIM_2 = 2;

aclnnStatus CheckNsaCmpAttnParam(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                 const char* inputLayout, const aclTensor* softmaxMaxOut,
                                 const aclTensor* softmaxSumOut, const aclTensor* attentionOutOut,
                                 const aclTensor* topkIndicesOut, const uint64_t* workspaceSize,
                                 aclOpExecutor** const executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(topkIndicesOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNsaCmpAttnShape(const aclTensor* query, const aclTensor* key, const aclTensor* value)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    auto qDim = qShape.GetDimNum();
    auto kDim = kShape.GetDimNum();
    auto vDim = vShape.GetDimNum();
    if (qDim != DIM_NUM_3 || kDim != DIM_NUM_3 || vDim != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "input shape error: expected all shapes to be 3D, but got qDim=%zu, kDim=%zu, vDim=%zu", qDim, kDim,
                vDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t axis = 0; axis < DIM_NUM_3; ++axis) {
        if (qShape[axis] <= 0 || kShape[axis] <= 0 || vShape[axis] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "input dimensions must be positive, but got query[%zu]=%ld, key[%zu]=%ld, value[%zu]=%ld", axis,
                    qShape[axis], axis, kShape[axis], axis, vShape[axis]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (kShape[0] != vShape[0] || kShape[1] != vShape[1]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "key and value must have the same T and N, but got key=(%ld, %ld), value=(%ld, %ld)", kShape[0],
                kShape[1], vShape[0], vShape[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qShape[2] != kShape[2] || kShape[2] < vShape[2]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "head dimensions must satisfy qD == kD and kD >= vD, but got qD=%ld, kD=%ld, vD=%ld", qShape[2],
                kShape[2], vShape[2]);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qShape[1] < kShape[1] || qShape[1] % kShape[1] != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "query head count must be a positive multiple of key head count, but got Nq=%ld, Nkv=%ld", qShape[1],
                kShape[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus NsaCmpAttnContiguous(const aclTensor*& query, const aclTensor*& key, const aclTensor*& value,
                                 const aclTensor*& attenMaskOptional, const aclTensor*& topkMaskOptional,
                                 aclOpExecutor* executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (topkMaskOptional) {
        topkMaskOptional = l0op::Contiguous(topkMaskOptional, executor);
        CHECK_RET(topkMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressAttentionVarLenScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaCompressAttentionVarLenScore);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnNsaCompressAttention(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaCompressAttention);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus Preprocess(const aclTensor*& query, const aclTensor*& key, aclOpExecutor* executor)
{
    // Query transpose from T,N1,D to N2,T,G,D
    // Only support TND now
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();

    FVector<int64_t, DIM_NUM_4> reshapedQueryShape = {qShape[DIM_0], kShape[DIM_1], qShape[DIM_1] / kShape[DIM_1],
                                                      qShape[DIM_2]}; // T,N2,G,D
    query =
        l0op::Reshape(query, executor->AllocIntArray(reshapedQueryShape.data(), reshapedQueryShape.size()), executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t, DIM_NUM_4> transedQueryShape = {1, 0, 2, 3}; // N2,T,G,D
    auto perm = executor->AllocIntArray(transedQueryShape.data(), transedQueryShape.size());
    query = l0op::Transpose(query, perm, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus Postprocess(const aclTensor*& attenOut, const aclTensor*& topkIndicesOut, aclOpExecutor* executor)
{
    // AttenOut transpose from N2,T,G,D2 to T,N1,D2
    Shape attenOutShape = attenOut->GetViewShape(); // N2,T,G,D2

    FVector<int64_t, DIM_NUM_4> transedShape = {1, 0, 2, 3}; // T,N2,G,D2
    auto perm = executor->AllocIntArray(transedShape.data(), transedShape.size());
    attenOut = l0op::Transpose(attenOut, perm, executor);
    CHECK_RET(attenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t, DIM_NUM_3> reshapedShape = {attenOutShape[DIM_1], attenOutShape[DIM_0] * attenOutShape[DIM_2],
                                                 attenOutShape[DIM_3]}; // T,N1,D2
    attenOut = l0op::Reshape(attenOut, executor->AllocIntArray(reshapedShape.data(), reshapedShape.size()), executor);
    CHECK_RET(attenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // TopkIndicesOut transpose from N2,T,select_count to T,N2,select_count
    FVector<int64_t, DIM_NUM_3> transedSelShape = {1, 0, 2};
    perm = executor->AllocIntArray(transedSelShape.data(), transedSelShape.size());
    topkIndicesOut = l0op::Transpose(topkIndicesOut, perm, executor);
    CHECK_RET(topkIndicesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attenMaskOptional,
    const aclTensor* topkMaskOptional, const aclIntArray* actualSeqQLenOptional,
    const aclIntArray* actualCmpSeqKvLenOptional, const aclIntArray* actualSelSeqKvLenOptional, double scaleValue,
    int64_t headNum, char* inputLayout, int64_t sparseMode, int64_t compressBlockSize, int64_t compressStride,
    int64_t selectBlockSize, int64_t selectBlockCount, const aclTensor* softmaxMaxOut, const aclTensor* softmaxSumOut,
    const aclTensor* attentionOutOut, const aclTensor* topkIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    CHECK_RET(CheckNsaCmpAttnParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
                                   topkIndicesOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnNsaCompressAttentionVarLenScore,
                   DFX_IN(query, key, value, attenMaskOptional, topkMaskOptional, actualSeqQLenOptional,
                          actualCmpSeqKvLenOptional, actualSelSeqKvLenOptional, scaleValue, headNum, inputLayout,
                          sparseMode, compressBlockSize, compressStride, selectBlockSize, selectBlockCount),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, attentionOutOut, topkIndicesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && topkIndicesOut->IsEmpty() &&
        attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckNsaCmpAttnShape(query, key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    CHECK_RET(NsaCmpAttnContiguous(query, key, value, attenMaskOptional, topkMaskOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(Preprocess(query, key, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0NsaCompressAttentionOuts = l0op::NsaCompressAttention(
        query, key, value, attenMaskOptional, actualSeqQLenOptional, actualCmpSeqKvLenOptional,
        actualSelSeqKvLenOptional, topkMaskOptional, scaleValue, headNum, inputLayout, sparseMode, compressBlockSize,
        compressStride, selectBlockSize, selectBlockCount, l0Executor);

    auto l0SoftmaxMaxOut = l0NsaCompressAttentionOuts[0];
    auto l0SoftmaxSumOut = l0NsaCompressAttentionOuts[1];
    auto l0AttentionOutOut = l0NsaCompressAttentionOuts[2];
    auto l0TopkIndicesOutOut = l0NsaCompressAttentionOuts[3];

    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr ||
        l0TopkIndicesOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut or l0TopkIndicesOutOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    CHECK_RET(Postprocess(l0AttentionOutOut, l0TopkIndicesOutOut, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult3 = l0op::ViewCopy(l0TopkIndicesOutOut, topkIndicesOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressAttentionGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attenMaskOptional,
    const aclTensor* topkMaskOptional, const aclIntArray* actualSeqQLenOptional,
    const aclIntArray* actualCmpSeqKvLenOptional, const aclIntArray* actualSelSeqKvLenOptional, double scaleValue,
    int64_t headNum, char* inputLayout, int64_t sparseMode, int64_t compressBlockSize, int64_t compressStride,
    int64_t selectBlockSize, int64_t selectBlockCount, const aclTensor* softmaxMaxOut, const aclTensor* softmaxSumOut,
    const aclTensor* attentionOutOut, const aclTensor* topkIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    CHECK_RET(CheckNsaCmpAttnParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
                                   topkIndicesOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnNsaCompressAttention,
                   DFX_IN(query, key, value, attenMaskOptional, topkMaskOptional, actualSeqQLenOptional,
                          actualCmpSeqKvLenOptional, actualSelSeqKvLenOptional, scaleValue, headNum, inputLayout,
                          sparseMode, compressBlockSize, compressStride, selectBlockSize, selectBlockCount),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, attentionOutOut, topkIndicesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && topkIndicesOut->IsEmpty() &&
        attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckNsaCmpAttnShape(query, key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    CHECK_RET(NsaCmpAttnContiguous(query, key, value, attenMaskOptional, topkMaskOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(Preprocess(query, key, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0NsaCompressAttentionOuts = l0op::NsaCompressAttention(
        query, key, value, attenMaskOptional, actualSeqQLenOptional, actualCmpSeqKvLenOptional,
        actualSelSeqKvLenOptional, topkMaskOptional, scaleValue, headNum, inputLayout, sparseMode, compressBlockSize,
        compressStride, selectBlockSize, selectBlockCount, l0Executor);

    auto l0SoftmaxMaxOut = l0NsaCompressAttentionOuts[0];
    auto l0SoftmaxSumOut = l0NsaCompressAttentionOuts[1];
    auto l0AttentionOutOut = l0NsaCompressAttentionOuts[2];
    auto l0TopkIndicesOutOut = l0NsaCompressAttentionOuts[3];

    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr ||
        l0TopkIndicesOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut or l0TopkIndicesOutOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    CHECK_RET(Postprocess(l0AttentionOutOut, l0TopkIndicesOutOut, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult3 = l0op::ViewCopy(l0TopkIndicesOutOut, topkIndicesOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
}
#endif
