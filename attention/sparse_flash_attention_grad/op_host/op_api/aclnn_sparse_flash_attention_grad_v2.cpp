/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_flash_attention_grad_v2.h"

#include "opdev/platform.h"
#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *sparseIndices, const aclTensor *dOut,
    const aclTensor *out, const aclTensor *softmaxMax, const aclTensor *softmaxSum, const aclTensor *sinksOptional,
    const aclTensor *valueOptional, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional, const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    double scaleValue, int64_t sparseBlockSize, char *layoutOptional, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, bool deterministic, const aclTensor *dQueryOut, const aclTensor *dKeyOut,
    const aclTensor *dValueOutOptional, const aclTensor *dQueryRopeOutOptional, const aclTensor *dKeyRopeOutOptional,
    const aclTensor *dSinksOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseFlashAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream);

// V2 接口：sinks 传了则必须给 dSinks 输出（传了 sinks 却无 dSinks 会拒绝）；
// 不传 sinks（无 sink 路径）时 dSinks 可为 nullptr 或空占位 tensor。
// 位置参数顺序与 V1 签名完全一致，仅在 softmaxSum 后插 sinks、dKeyRopeOutOptional 后插
// dSinksOptional，复用同一 OpDef 与同一 inner aclnn。
aclnnStatus aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparseIndices,
    const aclTensor *dOut, const aclTensor *out, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *sinks, // oss-sink 输入 [N1] FP32，OPTIONAL；传了则必须给 dSinks
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, double scaleValue, int64_t sparseBlockSize,
    char *layoutOptional, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic,
    const aclTensor *dQueryOut, const aclTensor *dKeyOut, const aclTensor *dValueOut,
    const aclTensor *dQueryRopeOutOptional, const aclTensor *dKeyRopeOutOptional,
    const aclTensor *dSinksOptional, // sink 梯度 [N1] FP32，OPTIONAL；sinks 为空时可空占位
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 不强制 sinks/dSinks 严格成对：host tiling 已把 nullptr/空 shape 的 sinks
    // 统一识别为 hasSinks=false（sparse_flash_attention_grad_tiling_bs1_regbase.cpp），此时内核
    // 走无 sink 路径、不写 dSinks，故 dSinks 允许为 nullptr 或空占位 tensor（shape {0}，PTA 无
    // sink 时返回的占位输出）。仅保留危险方向校验：传了 sinks 却没给 dSinks 输出 buffer。
    if (sinks != nullptr && dSinksOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The sinks input of aclnnSparseFlashAttentionGradV2 requires a corresponding dSinks output.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sinks != nullptr && op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "The sinks input of aclnnSparseFlashAttentionGradV2 is only supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
        query, key, sparseIndices, dOut, out, softmaxMax, softmaxSum, sinks, value, actualSeqLengthsQueryOptional,
        actualSeqLengthsKvOptional, queryRopeOptional, keyRopeOptional, scaleValue, sparseBlockSize, layoutOptional,
        sparseMode, preTokens, nextTokens, deterministic, dQueryOut, dKeyOut, dValueOut, dQueryRopeOutOptional,
        dKeyRopeOutOptional, dSinksOptional, workspaceSize, executor);
}

aclnnStatus aclnnSparseFlashAttentionGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    return aclnnInnerSparseFlashAttentionGrad(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
