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
 * \file moe_finalize_routing_v2_grad.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_GRAD_H_
#define OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> MoeFinalizeRoutingV2Grad(
    const aclTensor *grad_y, const aclTensor *expanded_row_idx, const aclTensor *expanded_x, const aclTensor *scales,
    const aclTensor *expert_idx, const aclTensor *bias, int64_t drop_pad_mode, int64_t active_num, int64_t expert_num,
    int64_t expert_capacity, const aclTensor *grad_expanded_x_out, const aclTensor *grad_scales_out,
    aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_GRAD_H_
