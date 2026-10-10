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
#include "causal_conv1d.h"
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
OP_TYPE_REGISTER(CausalConv1d);

bool CausalConv1d(const aclTensor* x, const aclTensor* weight, aclTensor* convStates, const aclTensor* bias,
                  const aclTensor* queryStartLoc, const aclTensor* cacheIndices, const aclTensor* initialStateMode,
                  const aclTensor* numAcceptedTokens, const char* activation, int64_t nullBlockId, aclTensor* y,
                  aclOpExecutor* executor)
{
    L0_DFX(CausalConv1d, x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
           activation, nullBlockId, y, convStates);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CausalConv1d,
        OP_INPUT(x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens),
        OP_OUTPUT(convStates, y), OP_ATTR(activation, nullBlockId));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CausalConv1d ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
