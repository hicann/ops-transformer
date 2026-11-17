/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MSA_INDEX_SCORE_H_
#define OP_API_INC_MSA_INDEX_SCORE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMsaIndexScore的第一段接口，根据具体的计算流程，计算workspace大小。
 *  按 layout_query="TND"、pool_block_size=128 执行。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* startLoc, const aclTensor* blockTableOptional,
    const aclTensor* queryScaleOptional, const aclTensor* keyScaleOptional, const aclTensor* attenMaskOptional,
    const aclTensor* actualSeqQlenOptional, const aclTensor* actualSeqKlenOptional, char* layoutKeyOptional,
    int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, int64_t returnChoice, int64_t topk,
    int64_t compressFactor, const aclTensor* scoreOptional, const aclTensor* sparseIndicesOptional,
    const aclTensor* sparseValuesOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnMsaIndexScore的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnMsaIndexScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MSA_INDEX_SCORE_H_
