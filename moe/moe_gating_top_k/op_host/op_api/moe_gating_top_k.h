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
 * \file moe_gating_top_k.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_MOE_GATING_TOP_K_H_
#define OP_API_INC_LEVEL0_MOE_GATING_TOP_K_H_

#include "opdev/op_executor.h"

namespace l0op {
struct MoeGatingTopKResult {
    const aclTensor* y;
    const aclTensor* expertIdx;
    const aclTensor* outOut;
};

MoeGatingTopKResult MoeGatingTopK(const aclTensor* x, const aclTensor* bias, const aclTensor* input_ids,
                                  const aclTensor* tid2eid, int64_t k, int64_t k_group, int64_t group_count,
                                  int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag,
                                  double routed_scaling_factor, double eps, const aclTensor* y_out,
                                  const aclTensor* expert_idx_out, const aclTensor* out_out, aclOpExecutor* executor);
} // namespace l0op
#endif // OP_API_INC_LEVEL0_MOE_GATING_TOP_K_H_
