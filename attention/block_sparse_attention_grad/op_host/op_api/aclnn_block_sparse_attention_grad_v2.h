/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_V2_H_
#define ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_V2_H_

#include "aclnn_block_sparse_attention_grad.h"

#ifdef __cplusplus
extern "C" {
#endif

// quantMode: 0=FP16/BF16 (all six scales must be nullptr), 5=HiF8 per-tensor.
// Mode 5 requires six FLOAT32 [1] scales, BF16 attentionOut and BF16 gradients.
// dsQuantScaleOptional > 0: static dS quantization; == 0: task-local dynamic dS.
// P scale remains required; no default scale is introduced by V2.
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
    const aclTensor* dout, const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* attentionOut, const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* blockShapeOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional,
    const aclTensor* qDequantScaleOptional, const aclTensor* kDequantScaleOptional,
    const aclTensor* vDequantScaleOptional, const aclTensor* doDequantScaleOptional,
    const aclTensor* pQuantScaleOptional, const aclTensor* dsQuantScaleOptional, char* qInputLayout,
    char* kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, int64_t quantMode, aclTensor* dq, aclTensor* dk, aclTensor* dv, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradV2(void* workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor* executor,
                                                                                   const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_V2_H_
