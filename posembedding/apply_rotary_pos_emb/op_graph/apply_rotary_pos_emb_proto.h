/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_APPLY_ROTARY_POSEMB_OPS_H_
#define OPS_OP_PROTO_INC_APPLY_ROTARY_POSEMB_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 * @li query: A tensor for transformer query. Must be one of the following types: float16, float32, bfloat16.
 * @li key: A tensor for transformer key. Must be one of the following types: float16, float32, bfloat16.
 * @li cos: A tensor for rotary position embedding cos. Must be one of the following types: float16, float32, bfloat16.
 * @li sin: A tensor for rotary position embedding sin. Must be one of the following types: float16, float32, bfloat16.
 * @par Attributes:
 * @li layout: Optional.Explanation input format. 1-"BSND" 2-"SBND" 3-"BNSD", 4-"TND". Defaults to 1.
 * @li rotary_mode: Optional.Explanation rotary mode. Support "half","interleave","quarter". Defaults to "half".
 * @par Outputs:
 * @li query: A tensor for transformer query. Has the same shape as "query".
 * @li key: A tensor for transformer key. Has the same shape as "key".
 * @attention Constraints:
 * @li Inputs  tensor of query, key, cos, sin must be 4D.
 * @li query, key must be the same except for the N dimension, the shape of sin and cos must be the same.
 * @li The B dimension of cos and sin must be consistent with query and key or equal to 1.
 * @li The S dimension of query, key, sin, cos must be the same.
 * @li The N dimension of cos and sin must be equal to 1.
 * @li The D dimension of query, key, sin, cos must be the same and not greater than 1024.
 * @li The dtype of input tensor query, key, sin, cos must be the same.
 * @li When the rotary_mode is "half" or "interleave", the D dimension must be divisible by 2.
 * when the rotary_mode is "quarter", the D dimension must be divisible by 4
 * @li Atlas inference products and Atlas training products:
    - support types: float16, float32.
    - support layout: not support.
    - support rotary_mode: not support.
 * @li Atlas A2 products and Atlas A3 training products:
    - support types: float16, float32.
    - support layout: 1.
    - support rotary_mode: not support.
 * @li Ascend 950PR&950DT products
    - support types: float16, float32, bfloat16.
    - support layout: 1, 2, 3, 4.
    - support rotary_mode: "half","interleave","quarter".
 */
REG_OP(ApplyRotaryPosEmb)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(layout, Int, 1)
    .ATTR(rotary_mode, String, "half")
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(ApplyRotaryPosEmb)

} // namespace ge

#endif
