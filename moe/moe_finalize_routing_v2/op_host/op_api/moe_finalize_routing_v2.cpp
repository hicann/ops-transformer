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
 * \file moe_finalize_routing_v2.cpp
 * \brief
 */

#include "moe_finalize_routing_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeFinalizeRoutingV2);
const aclTensor* MoeFinalizeRoutingV2(const aclTensor* expanded_x, const aclTensor* expanded_row_idx,
                                      const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                      const aclTensor* scales, const aclTensor* expert_idx, const aclTensor* x,
                                      const aclTensor* alpha1, const aclTensor* alpha2, const aclTensor* v,
                                      int64_t drop_pad_mode, const aclIntArray* zeroExpertRange,
                                      const aclIntArray* copyExpertRange, const aclIntArray* constantExpertRange,
                                      int64_t k, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(MoeFinalizeRoutingV2, expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, x, alpha1, alpha2, v,
           drop_pad_mode, zeroExpertRange, copyExpertRange, constantExpertRange, k, out);

    auto y = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    // 如果输出是空，直接不走算子逻辑
    if (!y->IsEmpty()) {
        if (Ops::Transformer::AclnnUtil::IsRegbase()) {
            auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
                MoeFinalizeRoutingV2,
                OP_INPUT(expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, x, alpha1, alpha2, v),
                OP_OUTPUT(y), OP_ATTR(drop_pad_mode, zeroExpertRange, copyExpertRange, constantExpertRange, k));
            OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                                 "MoeFinalizeRoutingV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        } else {
            auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
                MoeFinalizeRoutingV2, OP_INPUT(expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx),
                OP_OUTPUT(y), OP_ATTR(drop_pad_mode, zeroExpertRange, copyExpertRange, constantExpertRange, k));
            OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                                 "MoeFinalizeRoutingV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        }
    }
    return y;
}
} // namespace l0op
