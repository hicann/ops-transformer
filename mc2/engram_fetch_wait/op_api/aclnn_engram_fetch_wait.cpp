/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_engram_fetch_wait.cpp
 * \brief engram_fetch_wait算子API实现
 */

#include "aclnn/aclnn_base.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "aclnnInner_engram_fetch_wait.h"

using namespace op;

static bool CheckNotNull(const aclTensor* commContext, aclTensor* fetched)
{
    OP_CHECK_NULL(commContext, return false);
    OP_CHECK_NULL(fetched, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* commContext, aclTensor* fetched)
{
    CHECK_RET(CheckNotNull(commContext, fetched), ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchWaitGetWorkspaceSize(const aclTensor* commContext, aclTensor* fetched,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto retParam = CheckParams(commContext, fetched);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = aclnnInnerEngramFetchWaitGetWorkspaceSize(commContext, fetched, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetchWait(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerEngramFetchWait(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
