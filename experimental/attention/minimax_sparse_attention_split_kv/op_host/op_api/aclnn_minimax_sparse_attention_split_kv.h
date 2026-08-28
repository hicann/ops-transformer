/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MINIMAX_SPARSE_ATTENTION_SPLIT_KV_H_
#define ACLNN_MINIMAX_SPARSE_ATTENTION_SPLIT_KV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// blockTable: paged KV cache [batch, maxBlocksPerBatch]; pass nullptr for contiguous
// dense K/V matching query + inputLayout.
// inputLayout: "TND" [T, N, D], "BNSD" [B, N, S, D], "BSND" [B, S, N, D].
// nullptr / empty defaults to "TND".
// softmaxLse: fp32 [T, N, 1] (TND), [B, N, S, 1] (BNSD) or [B, S, N, 1] (BSND)
// when softmaxLseFlag is true; ignored (may be nullptr) when false.
__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockTable,
    const aclTensor *k2qRowPtr, const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv, int64_t numKeyValueHeads, double scaleValue,
    int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag, const char *inputLayout,
    aclTensor *attentionOut, aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKv(void *workspace,
                                                                                      uint64_t workspaceSize,
                                                                                      aclOpExecutor *executor,
                                                                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_MINIMAX_SPARSE_ATTENTION_SPLIT_KV_H_
