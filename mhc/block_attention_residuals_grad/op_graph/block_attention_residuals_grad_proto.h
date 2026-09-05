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
 * \file block_attention_residuals_grad_proto.h
 * \brief block_attention_residuals_grad
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_BLOCK_ATTENTION_RESIDUALS_GRAD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_BLOCK_ATTENTION_RESIDUALS_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(BlockAttentionResidualsGrad)
    .INPUT(partial_block, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(block_res, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(proj_weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(norm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(grad_hidden_states, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(inv_norm, TensorType({DT_FLOAT}))
    .INPUT(probs, TensorType({DT_FLOAT}))
    .OUTPUT(grad_partial_block, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_block_res, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_proj_weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_norm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(valid_block_num, Int, 0)
    .OP_END_FACTORY_REG(BlockAttentionResidualsGrad)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_BLOCK_ATTENTION_RESIDUALS_GRAD_H_
