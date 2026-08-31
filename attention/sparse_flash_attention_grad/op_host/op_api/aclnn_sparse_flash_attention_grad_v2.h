/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H
#define ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparseIndices,
    const aclTensor *dOut, const aclTensor *out, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *sinks, const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, double scaleValue, int64_t sparseBlockSize,
    char *layoutOptional, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic,
    const aclTensor *dQueryOut, const aclTensor *dKeyOut, const aclTensor *dValueOut,
    const aclTensor *dQueryRopeOutOptional, const aclTensor *dKeyRopeOutOptional, const aclTensor *dSinksOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGradV2(void *workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor *executor,
                                                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H
