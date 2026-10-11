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
 * \file moe_init_routing_v2.cpp
 * \brief
 */

#include "moe_init_routing_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV2);

const std::array<const aclTensor*, 4> MoeInitRoutingV2(
    const aclTensor* x, const aclTensor* expert_idx, int64_t active_num, int64_t expert_capacity, int64_t expert_num,
    int64_t drop_pad_mode, int64_t expert_tokens_count_or_cumsum_flag, bool expert_tokens_before_capacity_flag,
    const aclTensor* expanded_x_out, const aclTensor* expanded_row_idx_out,
    const aclTensor* expert_tokens_count_or_cumsum_out, const aclTensor* expert_tokens_before_capacity_out,
    aclOpExecutor* executor)
{
    L0_DFX(MoeInitRoutingV2, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
           expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, expanded_x_out, expanded_row_idx_out,
           expert_tokens_count_or_cumsum_out, expert_tokens_before_capacity_out);

    auto expanded_x =
        executor->AllocTensor(expanded_x_out->GetViewShape(), expanded_x_out->GetDataType(), op::Format::FORMAT_ND);
    auto expanded_row_idx = executor->AllocTensor(expanded_row_idx_out->GetViewShape(),
                                                  expanded_row_idx_out->GetDataType(), op::Format::FORMAT_ND);

    const aclTensor* expert_tokens_count_or_cumsum = nullptr;
    if (expert_tokens_count_or_cumsum_out == nullptr) {
        // 可选输出需要使用空Tensor去占位
        op::Shape zeroShape;
        zeroShape.AppendDim(0);
        expert_tokens_count_or_cumsum = executor->AllocTensor(zeroShape, ge::DT_INT32, op::Format::FORMAT_ND);
    } else {
        expert_tokens_count_or_cumsum =
            executor->AllocTensor(expert_tokens_count_or_cumsum_out->GetViewShape(),
                                  expert_tokens_count_or_cumsum_out->GetDataType(), op::Format::FORMAT_ND);
    }

    const aclTensor* expert_tokens_before_capacity = nullptr;
    if (expert_tokens_before_capacity_out == nullptr) {
        // 可选输出需要使用空Tensor去占位
        op::Shape zeroShape;
        zeroShape.AppendDim(0);
        expert_tokens_before_capacity = executor->AllocTensor(zeroShape, ge::DT_INT32, op::Format::FORMAT_ND);
    } else {
        expert_tokens_before_capacity =
            executor->AllocTensor(expert_tokens_before_capacity_out->GetViewShape(),
                                  expert_tokens_before_capacity_out->GetDataType(), op::Format::FORMAT_ND);
    }

    if (!(expanded_x->IsEmpty() && expanded_row_idx->IsEmpty() && expert_tokens_count_or_cumsum->IsEmpty() &&
          expert_tokens_before_capacity->IsEmpty())) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            MoeInitRoutingV2, OP_INPUT(x, expert_idx),
            OP_OUTPUT(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity),
            OP_ATTR(active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                    expert_tokens_before_capacity_flag));

        const std::array<const aclTensor*, 4> errorReturn = {nullptr, nullptr, nullptr, nullptr};
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return errorReturn,
                                             "MoeInitRoutingV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }

    return {expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity};
}
} // namespace l0op
