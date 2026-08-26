/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_V2_H
#define ACLNN_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnSparseLightningIndexerGradKLLossV2
 * calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseLightningIndexerGradKLLossV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *queryIndex, const aclTensor *keyIndex,
    const aclTensor *weight, const aclTensor *sparseIndices, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclIntArray *actualSeqLengthsQueryOptional, const aclIntArray *actualSeqLengthsKeyOptional,
    const aclTensor *sinksOptional, double scaleValue, char *layout, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, bool deterministic, const aclTensor *dQueryIndex, const aclTensor *dKeyIndex,
    const aclTensor *dWeight, const aclTensor *loss, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of ACLNN_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_V2_H is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseLightningIndexerGradKLLossV2(void *workspace,
                                                                                           uint64_t workspaceSize,
                                                                                           aclOpExecutor *executor,
                                                                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_V2_H
