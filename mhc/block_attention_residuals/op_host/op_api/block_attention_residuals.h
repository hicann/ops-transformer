/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_BLOCK_ATTENTION_RESIDUALS_OP_HOST_OP_API_BLOCK_ATTENTION_RESIDUALS_H_
#define MHC_BLOCK_ATTENTION_RESIDUALS_OP_HOST_OP_API_BLOCK_ATTENTION_RESIDUALS_H_

#include <array>
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 3> BlockAttentionResiduals(const aclTensor *partialBlock, const aclTensor *blockRes,
                                                               const aclTensor *projWeight, const aclTensor *normWeight,
                                                               int64_t validBlockNum, double normEps, bool needBackward,
                                                               aclOpExecutor *executor);
} // namespace l0op

#endif // MHC_BLOCK_ATTENTION_RESIDUALS_OP_HOST_OP_API_BLOCK_ATTENTION_RESIDUALS_H_
