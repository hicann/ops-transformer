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
 * \file aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h
 * \brief
 */
#ifndef ACLNN_DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_H
#define ACLNN_DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_H

#include <string>
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDenseLightningIndexerKLLossGradMetadata(void *workspace,
                                                                                                uint64_t workspaceSize,
                                                                                                aclOpExecutor *executor,
                                                                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_H
