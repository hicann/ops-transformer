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
 * \file inplace_partial_rotary_mul_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_OPS_H_
#define OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply inplace partial rotary position embedding for a single tensor.
 * @par Inputs:
 * @li x: A tensor which rotary position embedding is applied inplace, format supports ND, and data type must be
 * float16, float or bfloat16.
 * @li cos: A tensor which is "cos" in rotary position embedding, format supports ND, data type must be the same as
 * "x", and shape must be the same as "sin".
 * @li sin: A tensor which is "sin" in rotary position embedding, format supports ND, data type must be the same as
 * "x", and shape must be the same as "cos".
 * @par Outputs:
 * x: A tensor which is the result of inplace rotary position embedding, format supports ND, data type must be
 * the same as input "x", and shape must be the same as input "x".
 * @par Attributes:
 * rotary_mode: An optional attribute of type int, specifying the mode of rotary position embedding. Currently only
 * supports interleave mode (rotary_mode=1). Defaults to 0.
 * partial_slice: An optional attribute of type list int, specifying the slice range [start, end) for partial rotary.
 * Defaults to [0, 0]. When sliceEnd equals sliceStart, no rotary position embedding is performed.
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the input "x". Under this representation, the shape constraints of each
 * parameter can be described as follows:
 * @li Non-contiguous tensors are not supported.
 * @li The D of "x" must not exceed 1024.
 * @li In interleave mode (rotary_mode=1), D of "x" must be a multiple of 2.
 * @li The last dimension sizes of "cos" and "sin" must be equal, and must equal the partialSlice length
 * (partialSlice[1] - partialSlice[0]).
 * @li partialSlice constraints: sliceStart >= 0, sliceEnd >= 0, sliceEnd <= D of "x",
 * sliceLength = sliceEnd - sliceStart >= 0.
 * @li Only interleave mode (rotary_mode=1) is supported.
 * @li Inplace execution: input "x" and output "x" share the same tensor, results are written back to input "x".
 * @li cos/sin shapes must satisfy broadcast relationship with "x".
 * @li On Ascend 950PR&950DT products: input "x" supports only BSND, B1ND, B11D, 111D layouts.
 * @li On Atlas A3 products and Atlas A2 products: input "x" supports only BS1D, B11D layouts.
 */
REG_OP(InplacePartialRotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(rotary_mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0, 0})
    .OP_END_FACTORY_REG(InplacePartialRotaryMul)

} // namespace ge

#endif
