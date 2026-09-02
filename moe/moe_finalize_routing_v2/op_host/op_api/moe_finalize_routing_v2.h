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
 * \file moe_finalize_routing_v2.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_H_
#define OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *MoeFinalizeRoutingV2(const aclTensor *expanded_x, const aclTensor *expanded_row_idx,
                                      const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                      const aclTensor *scales, const aclTensor *expert_idx, const aclTensor *x,
                                      const aclTensor *alpha1, const aclTensor *alpha2, const aclTensor *v,
                                      int64_t drop_pad_mode, const aclIntArray *zeroExpertRange,
                                      const aclIntArray *copyExpertRange, const aclIntArray *constantExpertRange,
                                      int64_t k, const aclTensor *out, aclOpExecutor *executor);
}
#endif // OP_API_INC_LEVEL0_MOE_FINALIZE_ROUTING_V2_H_
