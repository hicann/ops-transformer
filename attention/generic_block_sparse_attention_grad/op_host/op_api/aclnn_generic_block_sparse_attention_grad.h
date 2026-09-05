/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_H
#define ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First-stage API: compute workspace size for GenericBlockSparseAttentionGrad.
 * @domain aclnn_ops_train
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout, const aclTensor *out,
    const aclTensor *lse, const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, const aclTensor *metadata,
    const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclIntArray *blockShape,
    int64_t isPackedGqa, char *qInputLayout, char *kvInputLayout, double scaleValue, int64_t maskType,
    int64_t softmaxPrecision, int64_t windowSizeLeft, int64_t windowSizeRight, aclTensor *dQuery, aclTensor *dKey,
    aclTensor *dValue, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Second-stage API: execute GenericBlockSparseAttentionGrad.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(void *workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor *executor,
                                                                                        const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GENERIC_SPARSE_ATTENTION_GRAD_H
