/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_norm_rope_concat.cpp
 * \brief
 */
#include <string>
#include "graph/types.h"
#include "aclnn_norm_rope_concat.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNormRopeConcatGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* encoderQuery,
    const aclTensor* encoderKey, const aclTensor* encoderValue, const aclTensor* normQueryWeight,
    const aclTensor* normQueryBias, const aclTensor* normKeyWeight, const aclTensor* normKeyBias,
    const aclTensor* normAddedQueryWeight, const aclTensor* normAddedQueryBias, const aclTensor* normAddedKeyWeight,
    const aclTensor* normAddedKeyBias, const aclTensor* ropeSin, const aclTensor* ropeCos, int64_t normType,
    int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
    const aclTensor* queryOutput, const aclTensor* keyOutput, const aclTensor* valueOutput,
    const aclTensor* normQueryMean, const aclTensor* normQueryRstd, const aclTensor* normKeyMean,
    const aclTensor* normKeyRstd, const aclTensor* normAddedQueryMean, const aclTensor* normAddedQueryRstd,
    const aclTensor* normAddedKeyMean, const aclTensor* normAddedKeyRstd, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerNormRopeConcat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* encoderQuery,
    const aclTensor* encoderKey, const aclTensor* encoderValue, const aclTensor* normQueryWeight,
    const aclTensor* normQueryBias, const aclTensor* normKeyWeight, const aclTensor* normKeyBias,
    const aclTensor* normAddedQueryWeight, const aclTensor* normAddedQueryBias, const aclTensor* normAddedKeyWeight,
    const aclTensor* normAddedKeyBias, const aclTensor* ropeSin, const aclTensor* ropeCos, int64_t normType,
    int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
    const aclTensor* queryOutput, const aclTensor* keyOutput, const aclTensor* valueOutput,
    const aclTensor* normQueryMean, const aclTensor* normQueryRstd, const aclTensor* normKeyMean,
    const aclTensor* normKeyRstd, const aclTensor* normAddedQueryMean, const aclTensor* normAddedQueryRstd,
    const aclTensor* normAddedKeyMean, const aclTensor* normAddedKeyRstd, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerNormRopeConcatGetWorkspaceSize(
        query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
        normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin, ropeCos,
        normType, normAddedType, ropeType, concatOrder, eps, isTraining, queryOutput, keyOutput, valueOutput,
        normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
        normAddedKeyMean, normAddedKeyRstd, workspaceSize, executor);
}

aclnnStatus aclnnNormRopeConcat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerNormRopeConcat(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
