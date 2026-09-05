/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H
#define ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv, int64_t maskType,
    int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(void *workspace,
                                                                                                uint64_t workspaceSize,
                                                                                                aclOpExecutor *executor,
                                                                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H
