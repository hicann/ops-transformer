/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    int64_t chunkSize, const aclTensor *qKernelOut, const aclTensor *kKernelOut, const aclTensor *wKernelOut,
    const aclTensor *uKernelOut, const aclTensor *gKernelOut, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleComputeWy(void *workspace,
                                                                                     uint64_t workspaceSize,
                                                                                     aclOpExecutor *executor,
                                                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
