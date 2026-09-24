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
#ifndef OPS_OP_PROTO_INC_ROTARY_POSITION_EMBEDDING_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_ROTARY_POSITION_EMBEDDING_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backwards calculation of RotaryPositionEmbedding.
 * @par Inputs:
 * @li dy: A 3D or 4D tensor which represents the gradient of output "y" in RotaryPositionEmbedding, format
 * supports ND, and data type must be float16, float32 or bfloat16.
 * @li cos: A 3D or 4D tensor which is input "cos" in RotaryPositionEmbedding, format supports ND, data type must be
 * the same as "dy", and shape must be the same as "sin".
 * @li sin: A 3D or 4D tensor which is input "sin" in RotaryPositionEmbedding, format supports ND, data type must be
 * the same as "dy", and shape must be the same as "cos".
 * @li x: An optional 3D or 4D tensor which is input "x" in RotaryPositionEmbedding, format supports ND, data type
 * must be the same as "dy", and shape must be the same as "dy".
 * If "x" is nullptr, the output "dcos" and "dsin" is meaningless.
 * @par Outputs:
 * @li dx: A 3D or 4D Tensor which is the grad of input "x" in RotaryPositionEmbedding, format supports ND, data type
 * must be the same as "dy", and shape must be the same as "dy".
 * @li dcos: A 3D or 4D Tensor which is the grad of input "cos" in RotaryPositionEmbedding, format supports ND,
 * data type must be the same as "dy", and shape must be the same as "cos".
 * @li dsin: A 3D or 4D Tensor which is the grad of input "sin" in RotaryPositionEmbedding, format supports ND,
 * data type must be the same as "dy", and shape must be the same as "sin".
 * @par Attributes:
 * mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half",
 * 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0. Atlas A2 products and Atlas A3 training products
 * only support 0-"half" and 1-"interleave".
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the input "dy" (4D) or (T, N, D) for 3D TND layout. Under this
 * representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "dy", "cos", "sin", "x", "dx", "dcos" and "dsin" must be equal. For Ascend 950PR&950DT products, D
 * should be less or equal to 1024. For Atlas A2 products and Atlas A3 training products, D should be less or equal to
 * 896.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple
 * of 4.
 * @li B, S, N of "cos", "sin", "dcos" and "dsin" must meet one of the following four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, D).
 *  - B, S, N are the same as that of "dy", means the shape is (B, S, N, D).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "dy", means the shape is (B, 1, N,
 * D) or (B, S, 1, D).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "dy", means the shape is (1, 1, N, D),
 * (1, S, 1, D) or (B, 1, 1, D).
 */
REG_OP(RotaryPositionEmbeddingGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dcos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dsin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(RotaryPositionEmbeddingGrad)

} // namespace ge

#endif
