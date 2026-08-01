/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_BLOCK_SPARSE_ATTN_METADATA_H
#define ACLNN_QUANT_BLOCK_SPARSE_ATTN_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnQuantBlockSparseAttnMetadataGetWorkspaceSize(
    const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode,
    int64_t maskMode, const char *layoutQOptional, const char *layoutKvOptional,
    const char *layoutSparseIndicesOptional, const aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnQuantBlockSparseAttnMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_BLOCK_SPARSE_ATTN_METADATA_H
