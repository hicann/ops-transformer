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
 * \file apply_rotary_pos_emb_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_APPLY_ROTARY_POS_EMB_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_APPLY_ROTARY_POS_EMB_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backwards calculation of ApplyRotaryPosEmb.
 * @par Inputs:
 * @li grad_query_embed: A 3D or 4D tensor which represents the gradient of output "query" in ApplyRotaryPosEmb, format
 * supports ND, and data type must be float16, float32 or bfloat16.
 * @li grad_key_embed: A 3D or 4D tensor which represents the gradient of output "key" in ApplyRotaryPosEmb, format
 * supports ND, and data type must be float16, float32 or bfloat16.
 * @li cos: A 3D or 4D tensor which is input "cos" in ApplyRotaryPosEmb, format supports ND, and data type
 * must be float16, float32 or bfloat16.
 * @li sin: A 3D or 4D tensor which is input "sin" in ApplyRotaryPosEmb, format supports ND, and data type
 * must be float16, float32 or bfloat16.
 * @li query: A 3D or 4D tensor which is input "query" in ApplyRotaryPosEmb, format supports ND, and data type
 * must be float16, float32 or bfloat16.
 * @li key: A 3D or 4D tensor which is input "key" in ApplyRotaryPosEmb, format supports ND, and data type
 * must be float16, float32 or bfloat16.
 * If "query" and "key" is nullptr, the output "grad_cos" and "grad_sin" is meaningless.
 *
 * @par Outputs:
 * @li grad_query: A 3D or 4D tensor which is the grad of input "query" in ApplyRotaryPosEmb, format supports ND, data
type
 * must be same as "grad_query_embed", and shape must be same as "grad_query_embed".
 * @li grad_key: A 3D or 4D tensor which is the grad of input "key" in ApplyRotaryPosEmb, format supports ND, data type
 * must be same as "grad_key_embed", and shape must be same as "grad_key_embed".
 * @li grad_cos: A 3D or 4D tensor which is the grad of input "cos" in ApplyRotaryPosEmb, format supports ND, data type
 * must be same as "grad_query_embed", and shape must be same as "cos".
 * @li grad_sin: A 3D or 4D tensor which is the grad of input "sin" in ApplyRotaryPosEmb, format supports ND, data type
 * must be same as "grad_query_embed", and shape must be same as "sin".
 *
 * @par Attributes:
 * @li rotary_mode: An optional attribute of type string, specifying the mode of rotary position embedding, must be
"half".
 * Defaults to "half".
 * @li layout: An optional attribute of type int, specifying the input tensor of layout, must be 1(BSND), 2(SBND),
4(TND).
 * Defaults to 1.

 * @attention Constraints:
* Let (B, S, N, D) represents the shape of the input "grad_query_embed" and "grad_key_embed" (4D) or (T, N, D) for 3D
TND layout.
* Under this representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "grad_query_embed", "grad_key_embed", "cos", "sin", "query", "key", "grad_cos", "grad_sin" must be
equal.
 * D should be less or equal to 1024. And In half mode, D must be a multiple of 2.
 * @li The shape and of "grad_query_embed", "grad_query" and "query" must be same, the shape of "grad_key_embed",
"grad_key" and "key" must be same.
 * @li The data type and dimension nums of all inputs must be same.
 * @li For any "layout", the dimensions of "grad_query_embed" and "grad_query" must be same except for the N dimension.
 * @li The N dimension of "cos" and "sin" must be 1, the B dimension of "cos" and "sin" must be 1 or be same with
"grad_query_embed",
 * and other dimension of "cos" and "sin" must be same with "grad_query_embed".
 * @li The shapes of "cos", "sin", "grad_cos" and "grad_sin" must be same.
**/

REG_OP(ApplyRotaryPosEmbGrad)
    .INPUT(grad_query_embed, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(grad_key_embed, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(rotary_mode, String, "half")
    .ATTR(layout, Int, 1)
    .OP_END_FACTORY_REG(ApplyRotaryPosEmbGrad)

} // namespace ge

#endif
