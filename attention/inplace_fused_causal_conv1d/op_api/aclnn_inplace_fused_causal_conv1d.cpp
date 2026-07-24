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
#include "aclnn_inplace_fused_causal_conv1d.h"

using namespace op;

namespace {

aclnnStatus InplaceFusedCausalConv1dCommonProcess(
    aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *initialStateMode, const aclTensor *bias,
    const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
    const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
    const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId, int64_t runMode, int64_t maxQueryLen,
    int64_t residualConnection, int64_t blockSize, int64_t convMode, uint64_t *workspaceSize, aclOpExecutor **executor)
{
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

    // Optional tensors: contiguous if non-null
    auto ensureContiguous = [&](const aclTensor *t) -> const aclTensor * {
        if (t == nullptr) {
            return nullptr;
        }
        return l0op::Contiguous(t, uniqueExecutor.get());
    };

    const aclTensor *queryStartLocFinal = ensureContiguous(queryStartLoc);
    const aclTensor *cacheIndicesFinal = ensureContiguous(cacheIndices);
    const aclTensor *initialStateModeFinal = ensureContiguous(initialStateMode);
    const aclTensor *biasFinal = ensureContiguous(bias);
    const aclTensor *numAcceptedTokensFinal = ensureContiguous(numAcceptedTokens);
    const aclTensor *numComputedTokensFinal = ensureContiguous(numComputedTokens);
    const aclTensor *blockIdxFirstFinal = ensureContiguous(blockIdxFirstScheduledToken);
    const aclTensor *blockIdxLastFinal = ensureContiguous(blockIdxLastScheduledToken);
    const aclTensor *initialStateIdxFinal = ensureContiguous(initialStateIdx);

    bool ok = l0op::InplaceFusedCausalConv1d(
        xFinal, weightFinal, convStatesFinal, queryStartLocFinal, cacheIndicesFinal, initialStateModeFinal, biasFinal,
        numAcceptedTokensFinal, numComputedTokensFinal, blockIdxFirstFinal, blockIdxLastFinal, initialStateIdxFinal,
        activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnInplaceFusedCausalConv1dGetWorkspaceSize(
    aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *initialStateMode, const aclTensor *bias,
    const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
    const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
    const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId, int64_t runMode, int64_t maxQueryLen,
    int64_t residualConnection, int64_t blockSize, int64_t convMode, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFusedCausalConv1d,
                   DFX_IN(x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens,
                          numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx),
                   DFX_OUT(convStates, x));
    return InplaceFusedCausalConv1dCommonProcess(
        x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens,
        numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, activationMode,
        padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnInplaceFusedCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFusedCausalConv1d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
