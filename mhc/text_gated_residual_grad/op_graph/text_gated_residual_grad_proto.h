/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_proto.h
 * \brief TextGatedResidualGrad graph operator registration.
 */

#ifndef OPS_OP_PROTO_INC_TEXT_GATED_RESIDUAL_GRAD_H_
#define OPS_OP_PROTO_INC_TEXT_GATED_RESIDUAL_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Computes gradients for the RMS normalization, mixed output and injection
 * branch of TextGatedResidual.
 * B is the batch size, S is the sequence length, C = channels, H is the hidden
 * size per channel, D = C * H, and R is the projection size. All tensors use ND
 * format. All FLOAT16 or BFLOAT16 inputs must have the same dtype as x.
 *
 * @par Inputs:
 * @li grad_mixed: Gradient of the mixed output. A 3D tensor of shape [B, S, H].
 *     Type supports FLOAT16 and BFLOAT16.
 * @li x: Forward input. A 3D tensor of shape [B, S, D]. Type supports FLOAT16
 *     and BFLOAT16.
 * @li w_down: Down-projection weight. A 2D tensor of shape [R, D]. Type supports
 *     FLOAT16 and BFLOAT16.
 * @li w_up: Up-projection weight. A 2D tensor of shape [D, R]. Type supports
 *     FLOAT16 and BFLOAT16.
 * @li inv_rms: Saved inverse RMS. A 3D tensor of shape [B, S, C]. Type is FLOAT32.
 * @li down_pre_act: Saved down-projection result before division by C. A 3D
 *     tensor of shape [B, S, R]. Type supports FLOAT16 and BFLOAT16.
 * @li up: Saved up-projection result. A 3D tensor of shape [B, S, D]. Type
 *     supports FLOAT16 and BFLOAT16.
 * @li grad_inject: Optional gradient of the injection output. A 3D tensor of
 *     shape [B, S, C]. Type supports FLOAT16 and BFLOAT16.
 * @li w_gate: Optional gate-projection weight. A 2D tensor of shape [C, D].
 *     Type supports FLOAT16 and BFLOAT16. When provided, grad_inject and inject
 *     are required.
 * @li gamma: Optional RMS weight. A 1D tensor of shape [D]. Type is FLOAT32.
 *     Normalization uses 1 + gamma; omitting gamma uses a scale of 1.
 * @li inject: Optional saved injection output. A 3D tensor of shape [B, S, C].
 *     Type supports FLOAT16 and BFLOAT16.
 *
 * @par Outputs:
 * @li dx: Gradient of x. A 3D tensor of shape [B, S, D], with the dtype of x.
 * @li dw_down: Gradient of w_down. A 2D tensor of shape [R, D], with the dtype
 *     of w_down.
 * @li dw_up: Gradient of w_up. A 2D tensor of shape [D, R], with the dtype of w_up.
 * @li dw_gate: Gradient of w_gate. When w_gate is provided, a 2D tensor of shape
 *     [C, D], with the dtype of w_gate. Otherwise, an empty tensor of shape [0].
 * @li dgamma: Gradient of gamma. When gamma is provided, a 1D tensor of shape
 *     [D], with type FLOAT32. Otherwise, an empty tensor of shape [0].
 *
 * @par Attributes:
 * @li channels: A required positive int attribute specifying the number of
 *     channels C. The last dimension of x must be divisible by channels.
 */
REG_OP(TextGatedResidualGrad)
    .INPUT(grad_mixed, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(w_down, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(w_up, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(inv_rms, TensorType({DT_FLOAT}))
    .INPUT(down_pre_act, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(up, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(grad_inject, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(w_gate, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(inject, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(channels, Int)
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dw_down, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dw_up, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dw_gate, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TextGatedResidualGrad)
} // namespace ge

#endif // OPS_OP_PROTO_INC_TEXT_GATED_RESIDUAL_GRAD_H_
