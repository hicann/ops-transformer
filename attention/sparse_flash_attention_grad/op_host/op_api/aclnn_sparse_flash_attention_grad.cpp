/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_flash_attention_grad.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* sparseIndices, const aclTensor* dOut,
    const aclTensor* out, const aclTensor* softmaxMax, const aclTensor* softmaxSum, const aclTensor* sinksOptional,
    const aclTensor* valueOptional, const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKvOptional, const aclTensor* queryRopeOptional, const aclTensor* keyRopeOptional,
    double scaleValue, int64_t sparseBlockSize, char* layoutOptional, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, bool deterministic, const aclTensor* dQueryOut, const aclTensor* dKeyOut,
    const aclTensor* dValueOutOptional, const aclTensor* dQueryRopeOutOptional, const aclTensor* dKeyRopeOutOptional,
    const aclTensor* dSinksOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerSparseFlashAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      aclrtStream stream);

// V1 接口：无 sinks/dSinks（5 输出），位置参数与商发 npu_sparse_flash_attention_grad 严格对齐。
// 转发 inner 时 sinks/dSinks 传 nullptr。
aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* sparseIndices,
    const aclTensor* dOut, const aclTensor* out, const aclTensor* softmaxMax, const aclTensor* softmaxSum,
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKvOptional,
    const aclTensor* queryRopeOptional, const aclTensor* keyRopeOptional, double scaleValue, int64_t sparseBlockSize,
    char* layoutOptional, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic,
    const aclTensor* dQueryOut, const aclTensor* dKeyOut, const aclTensor* dValueOut,
    const aclTensor* dQueryRopeOutOptional, const aclTensor* dKeyRopeOutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
        query, key, sparseIndices, dOut, out, softmaxMax, softmaxSum, nullptr /*sinks*/, value,
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, queryRopeOptional, keyRopeOptional, scaleValue,
        sparseBlockSize, layoutOptional, sparseMode, preTokens, nextTokens, deterministic, dQueryOut, dKeyOut,
        dValueOut, dQueryRopeOutOptional, dKeyRopeOutOptional, nullptr /*dSinks*/, workspaceSize, executor);
}

aclnnStatus aclnnSparseFlashAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    return aclnnInnerSparseFlashAttentionGrad(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
