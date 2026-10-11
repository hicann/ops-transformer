/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_update_expert.h"
#include <algorithm>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_update_expert.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(const aclTensor* expertIds, const aclTensor* eplbTable,
                                                 const aclTensor* expertScalesOptional,
                                                 const aclTensor* pruningThresholdOptional,
                                                 const aclTensor* activeMaskOptional, int64_t localRankId,
                                                 int64_t worldSize, int64_t balanceMode, aclTensor* balancedExpertIds,
                                                 aclTensor* balancedActiveMask, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(expertIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(eplbTable, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(balancedExpertIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(balancedActiveMask, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus ret = aclnnInnerMoeUpdateExpertGetWorkspaceSize(
        expertIds, eplbTable, expertScalesOptional, pruningThresholdOptional, activeMaskOptional, localRankId,
        worldSize, balanceMode, balancedExpertIds, balancedActiveMask, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeUpdateExpert(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerMoeUpdateExpert(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
