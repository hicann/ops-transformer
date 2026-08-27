/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "inplace_fused_causal_conv1d.h"
#include "aclnn_inplace_fused_causal_conv1d_v2.h"

using namespace op;

namespace {

aclnnStatus InplaceFusedCausalConv1dV2CommonProcess(
    aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *initialStateMode, const aclTensor *bias,
    const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
    const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
    const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId, int64_t runMode, int64_t maxQueryLen,
    int64_t residualConnection, int64_t blockSize, int64_t convMode, int64_t maxDraftTokens, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // Mandatory tensors must be checked before any dereference (CreateView / Contiguous).
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(convStates, return ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();

    // Handle non-contiguous x input via CreateView (dual shape descriptor, zero-copy).
    aclTensor *xFinal =
        uniqueExecutor->CreateView(x, x->GetViewShape(), x->GetStorageShape(), x->GetViewStrides(), x->GetViewOffset());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for x failed.");

    aclTensor *convStatesFinal =
        uniqueExecutor->CreateView(convStates, convStates->GetViewShape(), convStates->GetStorageShape(),
                                   convStates->GetViewStrides(), convStates->GetViewOffset());
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for convStates failed.");

    // weight must be contiguous
    const aclTensor *weightFinal = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weightFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

    // Optional tensors: contiguous if non-null.
    // CHECK_COND at each call site distinguishes "not provided" (t == nullptr, legal) from
    // "provided but Contiguous failed" (final == nullptr), so a genuine conversion failure is
    // never silently degraded into "input not provided".
    auto ensureContiguous = [&](const aclTensor *t) -> const aclTensor * {
        if (t == nullptr) {
            return nullptr;
        }
        return l0op::Contiguous(t, uniqueExecutor.get());
    };

    const aclTensor *queryStartLocFinal = ensureContiguous(queryStartLoc);
    CHECK_COND(queryStartLoc == nullptr || queryStartLocFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous queryStartLoc failed.");

    const aclTensor *cacheIndicesFinal = ensureContiguous(cacheIndices);
    CHECK_COND(cacheIndices == nullptr || cacheIndicesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous cacheIndices failed.");

    const aclTensor *initialStateModeFinal = ensureContiguous(initialStateMode);
    CHECK_COND(initialStateMode == nullptr || initialStateModeFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous initialStateMode failed.");

    const aclTensor *biasFinal = ensureContiguous(bias);
    CHECK_COND(bias == nullptr || biasFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous bias failed.");

    const aclTensor *numAcceptedTokensFinal = ensureContiguous(numAcceptedTokens);
    CHECK_COND(numAcceptedTokens == nullptr || numAcceptedTokensFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous numAcceptedTokens failed.");

    const aclTensor *numComputedTokensFinal = ensureContiguous(numComputedTokens);
    CHECK_COND(numComputedTokens == nullptr || numComputedTokensFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous numComputedTokens failed.");

    const aclTensor *blockIdxFirstFinal = ensureContiguous(blockIdxFirstScheduledToken);
    CHECK_COND(blockIdxFirstScheduledToken == nullptr || blockIdxFirstFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous blockIdxFirstScheduledToken failed.");

    const aclTensor *blockIdxLastFinal = ensureContiguous(blockIdxLastScheduledToken);
    CHECK_COND(blockIdxLastScheduledToken == nullptr || blockIdxLastFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous blockIdxLastScheduledToken failed.");

    const aclTensor *initialStateIdxFinal = ensureContiguous(initialStateIdx);
    CHECK_COND(initialStateIdx == nullptr || initialStateIdxFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Contiguous initialStateIdx failed.");

    bool ok = l0op::InplaceFusedCausalConv1d(
        xFinal, weightFinal, convStatesFinal, queryStartLocFinal, cacheIndicesFinal, initialStateModeFinal, biasFinal,
        numAcceptedTokensFinal, numComputedTokensFinal, blockIdxFirstFinal, blockIdxLastFinal, initialStateIdxFinal,
        activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, maxDraftTokens,
        uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize(
    aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *initialStateMode, const aclTensor *bias,
    const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
    const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
    const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId, int64_t runMode, int64_t maxQueryLen,
    int64_t residualConnection, int64_t blockSize, int64_t convMode, int64_t maxDraftTokens, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFusedCausalConv1dV2,
                   DFX_IN(x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens,
                          numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx),
                   DFX_OUT(convStates, x));
    return InplaceFusedCausalConv1dV2CommonProcess(
        x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens,
        numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, activationMode,
        padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, maxDraftTokens, workspaceSize,
        executor);
}

ACLNN_API aclnnStatus aclnnInplaceFusedCausalConv1dV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFusedCausalConv1dV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
