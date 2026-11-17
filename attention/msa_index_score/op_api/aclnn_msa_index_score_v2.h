/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MSA_INDEX_SCORE_V2_H_
#define OP_API_INC_MSA_INDEX_SCORE_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMsaIndexScoreV2的第一段接口，根据具体的计算流程，计算workspace大小。
 *  在原接口上暴露 layout_query 与 pool_block_size：pool_block_size 只允许 16 / 128，
 *  16 仅 Ascend 950、fp16 / bf16，走两级 maxpool；缺省按 layout_query="TND"、pool_block_size=128。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMsaIndexScoreV2GetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* startLoc, const aclTensor* blockTableOptional,
    const aclTensor* queryScaleOptional, const aclTensor* keyScaleOptional, const aclTensor* attenMaskOptional,
    const aclTensor* actualSeqQlenOptional, const aclTensor* actualSeqKlenOptional, char* layoutKeyOptional,
    int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, char* layoutQueryOptional, int64_t poolBlockSize,
    int64_t compressFactor, const aclTensor* score, uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnMsaIndexScoreV2的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnMsaIndexScoreV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MSA_INDEX_SCORE_V2_H_
