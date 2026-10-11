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
 * \file moe_finalize_routing_v2_grad.cpp
 * \brief
 */

#include "moe_finalize_routing_v2_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeFinalizeRoutingV2Grad);

const std::array<const aclTensor*, 2> MoeFinalizeRoutingV2Grad(
    const aclTensor* grad_y, const aclTensor* expanded_row_idx, const aclTensor* expanded_x, const aclTensor* scales,
    const aclTensor* expert_idx, const aclTensor* bias, int64_t drop_pad_mode, int64_t active_num, int64_t expert_num,
    int64_t expert_capacity, const aclTensor* grad_expanded_x_out, const aclTensor* grad_scales_out,
    aclOpExecutor* executor)
{
    L0_DFX(MoeFinalizeRoutingV2Grad, grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias, drop_pad_mode,
           active_num, expert_num, expert_capacity, grad_expanded_x_out, grad_scales_out);

    auto grad_expanded_x = executor->AllocTensor(grad_expanded_x_out->GetViewShape(),
                                                 grad_expanded_x_out->GetDataType(), op::Format::FORMAT_ND);
    auto grad_scales =
        executor->AllocTensor(grad_scales_out->GetViewShape(), grad_scales_out->GetDataType(), op::Format::FORMAT_ND);
    if (!(grad_expanded_x->IsEmpty() && grad_scales->IsEmpty())) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            MoeFinalizeRoutingV2Grad, OP_INPUT(grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias),
            OP_OUTPUT(grad_expanded_x, grad_scales), OP_ATTR(drop_pad_mode, active_num, expert_num, expert_capacity));

        const std::array<const aclTensor*, 2> errorReturn = {nullptr, nullptr};
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return errorReturn,
                                             "MoeFinalizeRoutingV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }

    return std::array<const aclTensor*, 2>{grad_expanded_x, grad_scales};
}
} // namespace l0op
