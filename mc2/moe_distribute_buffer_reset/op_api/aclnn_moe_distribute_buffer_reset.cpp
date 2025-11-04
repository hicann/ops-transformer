/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_distribute_buffer_reset.cpp
 * \brief
 */
#include "aclnn_moe_distribute_buffer_reset.h"
#include <algorithm>
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType:uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeBufferResetGetWorkspaceSize(const aclTensor *elasticInfo, const char *groupEp,
                                                                      int64_t epWorldSize, int64_t needSync,
                                                                      uint64_t *workspaceSize,
                                                                      aclOpExecutor **executor);
extern aclnnStatus aclnnInnerMoeDistributeBufferReset(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor *elasticInfo, const char *groupEp)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(elasticInfo, return false);
    if (groupEp == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name is Empty.");
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *elasticInfo, const char *groupEp)
{
    CHECK_RET(CheckNullStatus(elasticInfo, groupEp), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(groupEp, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Required groupEp name length in range (0, HCCL_GROUP_NAME_MAX), but it's %zu.",
                strnlen(groupEp, HCCL_GROUP_NAME_MAX));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeBufferResetGetWorkspaceSize(const aclTensor *elasticInfo, const char *groupEp,
                                                          int64_t epWorldSize, int64_t needSync,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor)

{
    auto retParam = CheckParams(elasticInfo, groupEp);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    return aclnnInnerMoeDistributeBufferResetGetWorkspaceSize(elasticInfo, groupEp, epWorldSize, needSync,
                                                              workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeBufferReset(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeDistributeBufferReset(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
