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
 * \file aclnn_engram_fetch_grad.cpp
 * \brief EngramFetchGrad 算子 aclnn 接口实现
 */

#include <algorithm>
#include "aclnn/aclnn_base.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "aclnnInner_engram_fetch_grad.h"

using namespace op;

static bool CheckNotNull(const aclTensor *commContext, const aclTensor *gradFetched, const aclTensor *perm,
                         const aclTensor *sendCounts, const aclTensor *recvCounts, const aclTensor *recvLocalEntry,
                         const aclTensor *numRecv, aclTensor *gradUniqueOut, aclTensor *uniqueLocalEntryOut,
                         aclTensor *numUniqueOut)
{
    OP_CHECK_NULL(commContext, return false);
    OP_CHECK_NULL(gradFetched, return false);
    OP_CHECK_NULL(perm, return false);
    OP_CHECK_NULL(sendCounts, return false);
    OP_CHECK_NULL(recvCounts, return false);
    OP_CHECK_NULL(recvLocalEntry, return false);
    OP_CHECK_NULL(numRecv, return false);
    OP_CHECK_NULL(gradUniqueOut, return false);
    OP_CHECK_NULL(uniqueLocalEntryOut, return false);
    OP_CHECK_NULL(numUniqueOut, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *commContext, const aclTensor *gradFetched, const aclTensor *perm,
                               const aclTensor *sendCounts, const aclTensor *recvCounts,
                               const aclTensor *recvLocalEntry, const aclTensor *numRecv, aclTensor *gradUniqueOut,
                               aclTensor *uniqueLocalEntryOut, aclTensor *numUniqueOut)
{
    CHECK_RET(CheckNotNull(commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv,
                           gradUniqueOut, uniqueLocalEntryOut, numUniqueOut),
              ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchGradGetWorkspaceSize(const aclTensor *commContext, const aclTensor *gradFetched,
                                                 const aclTensor *perm, const aclTensor *sendCounts,
                                                 const aclTensor *recvCounts, const aclTensor *recvLocalEntry,
                                                 const aclTensor *numRecv, aclTensor *gradUniqueOut,
                                                 aclTensor *uniqueLocalEntryOut, aclTensor *numUniqueOut,
                                                 int64_t numEntriesPerRank, int64_t commBufferSize,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto retParam = CheckParams(commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv,
                                gradUniqueOut, uniqueLocalEntryOut, numUniqueOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = aclnnInnerEngramFetchGradGetWorkspaceSize(
        commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv, numEntriesPerRank,
        commBufferSize, gradUniqueOut, uniqueLocalEntryOut, numUniqueOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetchGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerEngramFetchGrad(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
