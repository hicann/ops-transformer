/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "inplace_fused_causal_conv1d.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(InplaceFusedCausalConv1d);

bool InplaceFusedCausalConv1d(aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                              const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
                              const aclTensor *initialStateMode, const aclTensor *bias,
                              const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
                              const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
                              const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId,
                              int64_t runMode, int64_t maxQueryLen, int64_t residualConnection, int64_t blockSize,
                              int64_t convMode, int64_t maxDraftTokens, aclOpExecutor *executor)
{
    L0_DFX(InplaceFusedCausalConv1d, x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias,
           numAcceptedTokens, numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken,
           initialStateIdx, activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode,
           maxDraftTokens);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        InplaceFusedCausalConv1d,
        OP_INPUT(x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens,
                 numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx),
        OP_OUTPUT(convStates, x),
        OP_ATTR(activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode,
                maxDraftTokens));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "InplaceFusedCausalConv1d ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
