/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_SCALED_COSINE_ATTENTION_SCORE_H_
#define OPS_OP_PROTO_INC_SCALED_COSINE_ATTENTION_SCORE_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Fuses fp32 L2 statistics, cosine score and per-head logit scaling.
 *
 * @par Inputs:
 * @li query: [B,H,N,d], bfloat16/float16/float32.
 * @li key: [B,H,N,d], same dtype and shape as query.
 * @li scale: [H] or [H,1,1], float32 log-scale.
 *
 * @par Outputs:
 * @li attn_score: [B,H,N,N], same dtype as query.
 *
 * @par Attributes:
 * @li clamp_max: upper bound applied before exp, default 4.6052.
 * @li eps: positive value added to each squared norm, default 1e-12.
 */
REG_OP(ScaledCosineAttentionScore)
    .INPUT(query, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(key, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(attn_score, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(clamp_max, Float, 4.6052)
    .ATTR(eps, Float, 1e-12)
    .OP_END_FACTORY_REG(ScaledCosineAttentionScore)
} // namespace ge

#endif // OPS_OP_PROTO_INC_SCALED_COSINE_ATTENTION_SCORE_H_
