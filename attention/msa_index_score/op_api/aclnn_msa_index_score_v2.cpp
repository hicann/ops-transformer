/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstddef>
#include "aclnn_msa_index_score_v2.h"
#include "opdev/op_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

// aclnn_inner 生成的内部 ABI：参数 = def 全集。V2 与 base 两个公有包装共用同一个原型。
extern aclnnStatus aclnnInnerMsaIndexScoreGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* startLoc, const aclTensor* blockTableOptional,
    const aclTensor* queryScaleOptional, const aclTensor* keyScaleOptional, const aclTensor* attenMaskOptional,
    const aclTensor* actualSeqQlenOptional, const aclTensor* actualSeqKlenOptional, char* layoutKeyOptional,
    int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, char* layoutQueryOptional, int64_t poolBlockSize,
    int64_t returnChoice, int64_t topk, int64_t compressFactor, const aclTensor* scoreOptional,
    const aclTensor* sparseIndicesOptional, const aclTensor* sparseValuesOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerMsaIndexScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

ACLNN_API aclnnStatus aclnnMsaIndexScoreV2GetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* startLoc, const aclTensor* blockTableOptional,
    const aclTensor* queryScaleOptional, const aclTensor* keyScaleOptional, const aclTensor* attenMaskOptional,
    const aclTensor* actualSeqQlenOptional, const aclTensor* actualSeqKlenOptional, char* layoutKeyOptional,
    int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, char* layoutQueryOptional, int64_t poolBlockSize,
    int64_t compressFactor, const aclTensor* score, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (score == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    // 缺省归一：空 / 空串回落 TND；poolBlockSize 非正数回落 128（host tiling 再做 16/128 白名单校验）。
    char layoutQueryDefault[] = "TND";
    char* layoutQuery =
        (layoutQueryOptional == nullptr || layoutQueryOptional[0] == '\0') ? layoutQueryDefault : layoutQueryOptional;
    const int64_t poolBlockSizeNorm = (poolBlockSize <= 0) ? 128 : poolBlockSize;
    const aclnnStatus ret = aclnnInnerMsaIndexScoreGetWorkspaceSize(
        query, key, startLoc, blockTableOptional, queryScaleOptional, keyScaleOptional, attenMaskOptional,
        actualSeqQlenOptional, actualSeqKlenOptional, layoutKeyOptional, sparseMode, initBlocks, localBlocks,
        layoutQuery, poolBlockSizeNorm, 0, 16, compressFactor, score, nullptr, nullptr, workspaceSize, executor);
    return ret;
}

ACLNN_API aclnnStatus aclnnMsaIndexScoreV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    const aclnnStatus ret = aclnnInnerMsaIndexScore(workspace, workspaceSize, executor, stream);
    return ret;
}
