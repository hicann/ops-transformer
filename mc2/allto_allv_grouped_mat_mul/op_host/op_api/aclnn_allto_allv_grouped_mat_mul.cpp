/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include <algorithm>
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor* gmmX, const aclTensor* gmmWeight,
                                                                    const aclTensor* sendCountsTensorOptional,
                                                                    const aclTensor* recvCountsTensorOptional,
                                                                    const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                                                                    const char* group, int64_t epWorldSize,
                                                                    const aclIntArray* sendCounts, const aclIntArray* recvCounts, 
                                                                    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
                                                                    aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
                                                                    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);

// check nullptr
static bool CheckNullStatus(const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
                            const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                            const char* group, bool permuteOutFlag, aclTensor* gmmY, const aclTensor* mmYOptional, const aclTensor* permuteOutOptional)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(gmmY, return false);
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) && (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u",
        (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))), (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))), mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "Optional output flag does not match optional output ptr!");
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
                               const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                               const char* group, int64_t epWorldSize, bool permuteOutFlag, aclTensor* gmmY, aclTensor* mmYOptional, 
                               aclTensor* permuteOutOptional)
{
    (void)epWorldSize;      // Unused
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group,
              permuteOutFlag, gmmY, mmYOptional, permuteOutOptional), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor* gmmX, const aclTensor* gmmWeight,
                                                        const aclTensor* sendCountsTensorOptional, 
                                                        const aclTensor* recvCountsTensorOptional, 
                                                        const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                                                        const char* group, int64_t epWorldSize, 
                                                        const aclIntArray* sendCounts, const aclIntArray* recvCounts, 
                                                        bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, 
                                                        aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, 
        mmXOptional, mmWeightOptional, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, 
        mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag, gmmY,
        mmYOptional, permuteOutOptional, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif