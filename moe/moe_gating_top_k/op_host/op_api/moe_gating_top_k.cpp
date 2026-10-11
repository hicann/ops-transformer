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
 * \file moe_gating_top_k.cpp
 * \brief
 */

#include "moe_gating_top_k.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeGatingTopK);
MoeGatingTopKResult MoeGatingTopK(const aclTensor* x, const aclTensor* bias, const aclTensor* input_ids,
                                  const aclTensor* tid2eid, int64_t k, int64_t k_group, int64_t group_count,
                                  int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag,
                                  double routed_scaling_factor, double eps, const aclTensor* y_out,
                                  const aclTensor* expert_idx_out, const aclTensor* out_out, aclOpExecutor* executor)
{
    L0_DFX(MoeGatingTopK, x, bias, input_ids, tid2eid, k, k_group, group_count, group_select_mode, renorm, norm_type,
           out_flag, routed_scaling_factor, eps, y_out, expert_idx_out, out_out);

    auto y = executor->AllocTensor(y_out->GetViewShape(), y_out->GetDataType(), Format::FORMAT_ND);
    auto expert_idx =
        executor->AllocTensor(expert_idx_out->GetViewShape(), expert_idx_out->GetDataType(), Format::FORMAT_ND);
    auto out_out_alloc = executor->AllocTensor(out_out->GetViewShape(), out_out->GetDataType(), Format::FORMAT_ND);

    MoeGatingTopKResult emptyResult = {nullptr, nullptr, nullptr};
    if (!y->IsEmpty()) {
        float routedScalingFactorFloat = static_cast<float>(routed_scaling_factor);
        float epsFloat = static_cast<float>(eps);
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeGatingTopK, OP_INPUT(x, bias, input_ids, tid2eid),
                                               OP_OUTPUT(y, expert_idx, out_out_alloc),
                                               OP_ATTR(k, k_group, group_count, group_select_mode, renorm, norm_type,
                                                       out_flag, routedScalingFactorFloat, epsFloat));
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return emptyResult,
                                             "MoeGatingTopK ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
    MoeGatingTopKResult result = {y, expert_idx, out_out_alloc};
    return result;
}
} // namespace l0op
