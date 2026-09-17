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
 * \file fused_gdn_gating_proto.h
 * \brief Graph operator registration proto for FusedGdnGating.
 */

#ifndef FUSED_GDN_GATING_PROTO_H_
#define FUSED_GDN_GATING_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief FusedGdnGating operator.
 *
 * Computes a fused GDN gating operation:
 *   g = -exp(a_log) * softplus(a + dt_bias)
 *   beta_output = sigmoid(b)
 * where softplus(x) = log(1 + exp(beta * x)) / beta, reverting to the linear
 * branch softplus(x) = x when beta * x > threshold.
 *
 * @par Inputs:
 * Four inputs, including:
 * @li a_log: A 1-D Tensor of shape [num_heads]. dtype fp32/bf16/fp16.
 * @li a: A 2-D Tensor of shape [batch, num_heads]. dtype bf16/fp16.
 * @li b: A 2-D Tensor of shape [batch, num_heads]. dtype bf16/fp16 (same as a).
 * @li dt_bias: A 1-D Tensor of shape [num_heads]. dtype same as a_log.
 *
 * @par Attributes:
 * @li beta: A float. The softplus beta scaling factor. Default: 1.0.
 * @li threshold: A float. The softplus threshold for numerical stability. Default: 20.0.
 *
 * @par Outputs:
 * Two outputs, including:
 * @li g: A 3-D Tensor of shape [1, batch, num_heads]. dtype fp32.
 *        The gate output: -exp(a_log) * softplus(a + dt_bias).
 * @li beta_output: A 3-D Tensor of shape [1, batch, num_heads]. dtype same as a/b.
 *        The sigmoid(b) output.
 */
REG_OP(FusedGdnGating)
    .INPUT(a_log, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(a, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(b, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(dt_bias, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(g, TensorType({DT_FLOAT}))
    .OUTPUT(beta_output, TensorType({DT_BF16, DT_FLOAT16}))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(FusedGdnGating)

} // namespace ge

#endif // FUSED_GDN_GATING_PROTO_H_
