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
 * \file moe_init_routing_v2.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_MOE_INIT_ROUTING_V2_H_
#define OP_API_INC_LEVEL0_MOE_INIT_ROUTING_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 4> MoeInitRoutingV2(
    const aclTensor *x, const aclTensor *expert_idx, int64_t active_num, int64_t expert_capacity, int64_t expert_num,
    int64_t drop_pad_mode, int64_t expert_tokens_count_or_cumsum_flag, bool expert_tokens_before_capacity_flag,
    const aclTensor *expanded_x_out, const aclTensor *expanded_row_idx_out,
    const aclTensor *expert_tokens_count_or_cumsum_out, const aclTensor *expert_tokens_before_capacity_out,
    aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_MOE_INIT_ROUTING_V2_H_
