/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_chunk_gated_delta_rule_compute_wy.h"
#include "chunk_gated_delta_rule_compute_wy.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleComputeWyParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *qKernelOut = nullptr;
    const aclTensor *kKernelOut = nullptr;
    const aclTensor *wKernelOut = nullptr;
    const aclTensor *uKernelOut = nullptr;
    const aclTensor *gKernelOut = nullptr;
};

static aclnnStatus CheckNotNull(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    CHECK_COND(p.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(p.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(p.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(p.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(p.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(p.qKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "qKernelOut must not be nullptr.");
    CHECK_COND(p.kKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "kKernelOut must not be nullptr.");
    CHECK_COND(p.wKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wKernelOut must not be nullptr.");
    CHECK_COND(p.uKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "uKernelOut must not be nullptr.");
    CHECK_COND(p.gKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gKernelOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

// Every bound below mirrors Tiling4ChunkGatedDeltaRuleComputeWy in
// op_host/chunk_gated_delta_rule_compute_wy_tiling.cpp, which is the source of truth.
// Keep the two in step: without these checks an illegal shape reaches the kernel and
// is turned into an out-of-range GM offset instead of being rejected on the host.
static constexpr int64_t FIXED_CHUNK = 64;
static constexpr int64_t HEAD_DIM_ALIGN = 16;
static constexpr int64_t MAX_HEAD_DIM = 128;
static constexpr int64_t MAX_BATCH = 32;
static constexpr int64_t MAX_VALUE_HEADS = 64;
static constexpr size_t QKV_RANK = 4;
static constexpr size_t GATE_RANK = 3;

static aclnnStatus CheckDtype(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    CHECK_COND(p.q->GetDataType() == op::DataType::DT_FLOAT16 && p.k->GetDataType() == op::DataType::DT_FLOAT16 &&
                   p.v->GetDataType() == op::DataType::DT_FLOAT16 && p.beta->GetDataType() == op::DataType::DT_FLOAT16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v/beta must be FLOAT16.");
    CHECK_COND(p.g->GetDataType() == op::DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "g must be FLOAT32.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    const auto &qShape = p.q->GetViewShape();
    const auto &kShape = p.k->GetViewShape();
    const auto &vShape = p.v->GetViewShape();
    const auto &gShape = p.g->GetViewShape();
    const auto &betaShape = p.beta->GetViewShape();

    CHECK_COND(qShape.GetDimNum() == QKV_RANK && vShape.GetDimNum() == QKV_RANK, ACLNN_ERR_PARAM_INVALID,
               "q must be [B, T, Hk, K] and v must be [B, T, Hv, V].");
    CHECK_COND(gShape.GetDimNum() == GATE_RANK, ACLNN_ERR_PARAM_INVALID, "g must be [B, T, Hv].");
    CHECK_COND(kShape == qShape, ACLNN_ERR_PARAM_INVALID, "k must have the same shape as q.");
    CHECK_COND(betaShape == gShape, ACLNN_ERR_PARAM_INVALID, "beta must have the same shape as g.");

    const int64_t b = qShape.GetDim(0);
    const int64_t t = qShape.GetDim(1);
    const int64_t hk = qShape.GetDim(2);
    const int64_t kDim = qShape.GetDim(3);
    const int64_t hv = vShape.GetDim(2);
    const int64_t vDim = vShape.GetDim(3);

    CHECK_COND(b > 0 && t > 0 && hk > 0 && kDim > 0 && hv > 0 && vDim > 0, ACLNN_ERR_PARAM_INVALID,
               "all q/v dimensions must be positive.");
    CHECK_COND(vShape.GetDim(0) == b && vShape.GetDim(1) == t, ACLNN_ERR_PARAM_INVALID, "v must share B/T with q.");
    CHECK_COND(gShape.GetDim(0) == b && gShape.GetDim(1) == t && gShape.GetDim(2) == hv, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, T, Hv] matching q's B/T and v's Hv.");
    CHECK_COND(hv % hk == 0, ACLNN_ERR_PARAM_INVALID, "Hv must be divisible by Hk.");
    CHECK_COND(t % FIXED_CHUNK == 0, ACLNN_ERR_PARAM_INVALID, "T must be a multiple of 64.");
    CHECK_COND(kDim % HEAD_DIM_ALIGN == 0 && vDim % HEAD_DIM_ALIGN == 0, ACLNN_ERR_PARAM_INVALID,
               "K and V must be multiples of 16.");
    CHECK_COND(kDim <= MAX_HEAD_DIM && vDim <= MAX_HEAD_DIM, ACLNN_ERR_PARAM_INVALID, "K and V must be <= 128.");
    CHECK_COND(b <= MAX_BATCH, ACLNN_ERR_PARAM_INVALID, "B must be <= 32.");
    CHECK_COND(hv <= MAX_VALUE_HEADS, ACLNN_ERR_PARAM_INVALID, "Hv must be <= 64.");

    // Outputs are caller-allocated and written by ViewCopy, so a mismatch here is an
    // overflow of the user's buffer, not just a wrong result.
    const auto &qKernelShape = p.qKernelOut->GetViewShape();
    const auto &kKernelShape = p.kKernelOut->GetViewShape();
    const auto &wKernelShape = p.wKernelOut->GetViewShape();
    const auto &uKernelShape = p.uKernelOut->GetViewShape();
    const auto &gKernelShape = p.gKernelOut->GetViewShape();
    CHECK_COND(qKernelShape.GetDimNum() == QKV_RANK && kKernelShape.GetDimNum() == QKV_RANK &&
                   wKernelShape.GetDimNum() == QKV_RANK && uKernelShape.GetDimNum() == QKV_RANK &&
                   gKernelShape.GetDimNum() == GATE_RANK,
               ACLNN_ERR_PARAM_INVALID, "output rank is invalid.");
    const bool qkOk = qKernelShape.GetDim(0) == b && qKernelShape.GetDim(1) == hk && qKernelShape.GetDim(2) == t &&
                      qKernelShape.GetDim(3) == kDim && kKernelShape == qKernelShape;
    CHECK_COND(qkOk, ACLNN_ERR_PARAM_INVALID, "q_kernel/k_kernel must be [B, Hk, T, K].");
    CHECK_COND(wKernelShape.GetDim(0) == b && wKernelShape.GetDim(1) == hv && wKernelShape.GetDim(2) == t &&
                   wKernelShape.GetDim(3) == kDim,
               ACLNN_ERR_PARAM_INVALID, "w_kernel must be [B, Hv, T, K].");
    CHECK_COND(uKernelShape.GetDim(0) == b && uKernelShape.GetDim(1) == hv && uKernelShape.GetDim(2) == t &&
                   uKernelShape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "u_kernel must be [B, Hv, T, V].");
    CHECK_COND(gKernelShape.GetDim(0) == b && gKernelShape.GetDim(1) == hv && gKernelShape.GetDim(2) == t,
               ACLNN_ERR_PARAM_INVALID, "g_kernel must be [B, Hv, T].");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    CHECK_RET(CheckNotNull(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.chunkSize == FIXED_CHUNK, ACLNN_ERR_PARAM_INVALID, "chunk_size must be 64.");
    CHECK_RET(CheckDtype(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeInputsContiguous(ChunkGatedDeltaRuleComputeWyParams &p, aclOpExecutor *executor)
{
    CHECK_COND(DataContiguous(p.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous q failed.");
    CHECK_COND(DataContiguous(p.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous k failed.");
    CHECK_COND(DataContiguous(p.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous v failed.");
    CHECK_COND(DataContiguous(p.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous g failed.");
    CHECK_COND(DataContiguous(p.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous beta failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    int64_t chunkSize, const aclTensor *qKernelOut, const aclTensor *kKernelOut, const aclTensor *wKernelOut,
    const aclTensor *uKernelOut, const aclTensor *gKernelOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");

    ChunkGatedDeltaRuleComputeWyParams p{q,          k,          v,          g,          beta,      chunkSize,
                                         qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut};

    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleComputeWy, DFX_IN(q, k, v, g, beta),
                   DFX_OUT(qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    CHECK_RET(CheckParams(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(MakeInputsContiguous(p, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "MakeInputsContiguous failed.");

    auto result =
        l0op::ChunkGatedDeltaRuleComputeWy(p.q, p.k, p.v, p.g, p.beta, p.chunkSize, p.qKernelOut, p.kKernelOut,
                                           p.wKernelOut, p.uKernelOut, p.gKernelOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(l0op::ViewCopy(result[0], p.qKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], p.kKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[2], p.wKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[3], p.uKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[4], p.gKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleComputeWy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    CHECK_COND(stream != nullptr, ACLNN_ERR_PARAM_NULLPTR, "stream must not be nullptr.");
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleComputeWy);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkGatedDeltaRuleComputeWy launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
