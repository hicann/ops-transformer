/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ffn_to_attention.h"
#include "common/utils/op_mc2_def.h"
#include <algorithm>
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "aclnnInner_ffn_to_attention.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor *x, const aclTensor *sessionIds, const aclTensor *microBatchIds,
                            const aclTensor *tokenIds, const aclTensor *expertOffsets, const aclTensor *actualTokenNum,
                            const char *group)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(sessionIds, return false);
    OP_CHECK_NULL(microBatchIds, return false);
    OP_CHECK_NULL(tokenIds, return false);
    OP_CHECK_NULL(expertOffsets, return false);
    OP_CHECK_NULL(actualTokenNum, return false);
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnFFNToAttention", "group");
        return false;
    }

    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *sessionIds, const aclTensor *microBatchIds,
                               const aclTensor *tokenIds, const aclTensor *expertOffsets,
                               const aclTensor *actualTokenNum, const char *group)
{
    CHECK_RET(CheckNullStatus(x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, group),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnFFNToAttention", "group", std::to_string(strnlen(group, HCCL_GROUP_NAME_MAX)).c_str(),
            (std::string("The length of group must be less than ") + std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(const aclTensor *x, const aclTensor *sessionIds,
                                                const aclTensor *microBatchIds, const aclTensor *tokenIds,
                                                const aclTensor *expertOffsets, const aclTensor *actualTokenNum,
                                                const aclTensor *attnRankTable, const char *group, int64_t worldSize,
                                                const aclIntArray *tokenInfoTableShape,
                                                const aclIntArray *tokenDataShape, uint64_t *workspaceSize,
                                                aclOpExecutor **executor)
{
    auto retParam = CheckParams(x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    std::string groupName(group);
    aclnnStatus ret = aclnnInnerFFNToAttentionGetWorkspaceSize(
        x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, attnRankTable, groupName.data(),
        worldSize, tokenInfoTableShape, tokenDataShape, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnFFNToAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerFFNToAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
