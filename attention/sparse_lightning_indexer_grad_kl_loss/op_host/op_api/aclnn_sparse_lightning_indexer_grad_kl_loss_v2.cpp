/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "graph/types.h"
#include "aclnn_sparse_lightning_indexer_grad_kl_loss_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerSparseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *queryIndex, const aclTensor *keyIndex,
    const aclTensor *weight, const aclTensor *sparseIndices, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclIntArray *actualSeqLengthsQueryOptional, const aclIntArray *actualSeqLengthsKeyOptional,
    const aclTensor *sinksOptional, double scaleValue, char *layout, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, bool deterministic, const aclTensor *dQueryIndex, const aclTensor *dKeyIndex,
    const aclTensor *dWeight, const aclTensor *loss, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseLightningIndexerGradKLLoss(void *workspace, uint64_t workspaceSize,
                                                              aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSparseLightningIndexerGradKLLossV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *queryIndex, const aclTensor *keyIndex,
    const aclTensor *weight, const aclTensor *sparseIndices, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclIntArray *actualSeqLengthsQueryOptional, const aclIntArray *actualSeqLengthsKeyOptional,
    const aclTensor *sinksOptional, double scaleValue, char *layout, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, bool deterministic, const aclTensor *dQueryIndex, const aclTensor *dKeyIndex,
    const aclTensor *dWeight, const aclTensor *loss, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (sinksOptional != nullptr && op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "The sinks input of aclnnSparseLightningIndexerGradKLLossV2 is only supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return aclnnInnerSparseLightningIndexerGradKLLossGetWorkspaceSize(
        query, key, queryIndex, keyIndex, weight, sparseIndices, softmaxMax, softmaxSum, queryRopeOptional,
        keyRopeOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, sinksOptional, scaleValue, layout,
        sparseMode, preTokens, nextTokens, deterministic, dQueryIndex, dKeyIndex, dWeight, loss, workspaceSize,
        executor);
}

aclnnStatus aclnnSparseLightningIndexerGradKLLossV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream)
{
    return aclnnInnerSparseLightningIndexerGradKLLoss(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
