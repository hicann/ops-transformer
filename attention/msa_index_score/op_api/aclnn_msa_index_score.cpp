/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include "acl/acl.h"
#include "aclnn_msa_index_score.h"
#include "opdev/op_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

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

aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* startLoc, const aclTensor* blockTableOptional,
    const aclTensor* queryScaleOptional, const aclTensor* keyScaleOptional, const aclTensor* attenMaskOptional,
    const aclTensor* actualSeqQlenOptional, const aclTensor* actualSeqKlenOptional, char* layoutKeyOptional,
    int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, int64_t returnChoice, int64_t topk,
    int64_t compressFactor, const aclTensor* scoreOptional, const aclTensor* sparseIndicesOptional,
    const aclTensor* sparseValuesOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclDataType queryType = ACL_DT_UNDEFINED;
    if (query != nullptr && aclGetDataType(query, &queryType) == ACL_SUCCESS && queryType == ACL_INT8) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 原始参数表固定 layout_query="TND"、pool_block_size=128。
    return aclnnInnerMsaIndexScoreGetWorkspaceSize(
        query, key, startLoc, blockTableOptional, queryScaleOptional, keyScaleOptional, attenMaskOptional,
        actualSeqQlenOptional, actualSeqKlenOptional, layoutKeyOptional, sparseMode, initBlocks, localBlocks,
        const_cast<char*>("TND"), 128, returnChoice, topk, compressFactor, scoreOptional, sparseIndicesOptional,
        sparseValuesOptional, workspaceSize, executor);
}

aclnnStatus aclnnMsaIndexScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMsaIndexScore(workspace, workspaceSize, executor, stream);
}
