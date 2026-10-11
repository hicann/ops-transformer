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
 * \file moe_init_routing_v2_grad.cpp
 * \brief
 */

#include "moe_init_routing_v2_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV2Grad);

const aclTensor* MoeInitRoutingV2Grad(const aclTensor* grad_expanded_x, const aclTensor* expanded_row_idx,
                                      int64_t top_k, int64_t drop_pad_mode, int64_t active_num, const aclTensor* out,
                                      aclOpExecutor* executor)
{
    L0_DFX(MoeInitRoutingV2Grad, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num, out);

    auto grad_x = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), op::Format::FORMAT_ND);
    // 如果输出是空，直接不走算子逻辑
    if (!grad_x->IsEmpty()) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeInitRoutingV2Grad, OP_INPUT(grad_expanded_x, expanded_row_idx),
                                               OP_OUTPUT(grad_x), OP_ATTR(top_k, drop_pad_mode, active_num));

        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                             "MoeInitRoutingV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }

    return grad_x;
}

} // namespace l0op
