/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_V5_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_V5_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeDistributeCombineV5GetWorkspaceSize(
    const aclTensor* context, const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts, const aclTensor* expertScales,
    const aclTensor* tpSendCountsOptional, const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional, const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional, const aclTensor* sharedExpertXOptional, const aclTensor* elasticInfoOptional,
    const aclTensor* oriXOptional, const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional, const aclTensor* constExpertVOptional,
    const aclTensor* performanceInfoOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    const char* commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* xOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMoeDistributeCombineV5(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_DISTRIBUTE_COMBINE_V5_
