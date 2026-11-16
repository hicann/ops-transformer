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
 * \file engram_fused_weight_proto.h
 * \brief
 */
#ifndef ENGRAM_FUSED_WEIGHT_H_
#define ENGRAM_FUSED_WEIGHT_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief EngramFusedWeight. \n
* Element-wise fusion of the two Engram gating weight tensors: each pair of elements
* is promoted to float32 and multiplied, \n
* weight_fused = float(weight_hidden) * float(weight_embed). \n

* @par Inputs:
* Two inputs, including:
* @li weight_hidden: A Tensor of type float16 or bfloat16, with shape (hcMult, hiddenSize).
* The first weight (q_weight) of Engram gating. Format support ND. \n
* @li weight_embed: A Tensor of type float16 or bfloat16, with shape (hcMult, hiddenSize).
* The second weight (k_weight) of Engram gating. Format support ND. \n

* @par Outputs:
* weight_fused: A Tensor of type float32. Has the same shape as input "weight_hidden".
* Format support ND. \n

* @attention Constraints:
* @li "weight_hidden" and "weight_embed" must be 2D tensors with the same shape.
* @li "weight_hidden" and "weight_embed" must have the same dtype.
* @li The total number of input elements must be divisible by 256. A zero-element
* tensor is allowed and produces an empty output.
*/
REG_OP(EngramFusedWeight)
    .INPUT(weight_hidden, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight_embed, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(weight_fused, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(EngramFusedWeight)

} // namespace ge

#endif // ENGRAM_FUSED_WEIGHT_H_
