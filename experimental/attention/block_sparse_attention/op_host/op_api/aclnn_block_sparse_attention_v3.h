/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_SPARSE_ATTENTION_V3_H_
#define ACLNN_BLOCK_SPARSE_ATTENTION_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional, const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, char *qInputLayout,
    char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise,
    int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode,
    double dstTypeMax, aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3(void *workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor *executor,
                                                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_SPARSE_ATTENTION_V3_H_
