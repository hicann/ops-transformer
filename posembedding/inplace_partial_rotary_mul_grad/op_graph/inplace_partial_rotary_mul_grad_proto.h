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
 * \file inplace_partial_rotary_mul_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backwards calculation of InplacePartialRotaryMul.
 * @par Inputs:
 * @li dy: A 4D tensor which represents the gradient of output "y" in InplacePartialRotaryMul, updated
 * in-place to be the grad of input "x" in InplacePartialRotaryMul. Format supports ND, and data type must be float16,
 * float32 or bfloat16.
 * @li cos: A 4D tensor which is input "cos" in InplacePartialRotaryMul, format supports ND, data type must be
 * float16, float32 or bfloat16 and must be the same as "sin", and shape must be the same as "sin". The D of "cos"
 * must be equal to the slice length ("partial_slice"'s end - start).
 * @li sin: A 4D tensor which is input "sin" in InplacePartialRotaryMul, format supports ND, data type must be
 * float16, float32 or bfloat16 and must be the same as "cos", and shape must be the same as "cos". The D of "sin"
 * must be equal to the slice length ("partial_slice"'s end - start).
 * @par Outputs:
 * @li dy: A 4D Tensor, in-place updated from input "dy", which is the grad of input "x" in
 * InplacePartialRotaryMul. Format supports ND, data type and shape must be the same as input "dy".
 * @par Attributes:
 * @li rotary_mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be
 * 0-"half", 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * @li partial_slice: An optional attribute of type list int, specifying the slice range of the partial rotary position
 * embedding. The value is a list of two integers. Defaults to {0, 0}.
 * @attention Constraints:
 * @li This operator is only supported on Ascend 950 AI Processor.
 * @li Currently only interleave mode (rotary_mode=1) is supported. Other modes (half=0, quarter=2,
 * interleave-half=3) are not implemented yet.
 * Let (B, S, N, D) represents the shape of the input "dy". Under this
 * representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "dy" must be equal and less or equal to 1024. When the slice length is greater than 0, the D of
 * "cos", "sin" must be equal to the slice length ("partial_slice"'s end - start).
 * @li B, S, N of "cos", "sin" must broadcast to match the B, S, N of "dy". When the slice length is greater than 0,
 * the D of these tensors equals ("partial_slice"'s end - start). Specifically, B, S, N must meet one of the following
 * four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, end-start).
 *  - B, S, N are the same as that of "dy", means the shape is (B, S, N, end-start).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "dy", means the shape is (B, 1, N,
 * end-start) or (B, S, 1, end-start).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "dy", means the shape is (1, 1, N,
 * end-start), (1, S, 1, end-start) or (B, 1, 1, end-start).
 * @li Let the value of "partial_slice" be [start, end]. "start" must be in range [0, D], "end" must be in range
 * ["start", D]. When the slice length is greater than 0: in half, interleave and interleave-half mode,
 * ("end" - "start") must be a multiple of 2; in quarter mode, ("end" - "start") must be a multiple of 4.
 * @li Empty-input support: when the slice length is 0 ("partial_slice"'s start equals end, e.g. [0, 0]), or any of
 * "dy"/"cos"/"sin" is an empty tensor (some dim is 0), the operator performs a no-op: no rotary grad is computed and
 * the output equals the input (in-place).
 */

REG_OP(InplacePartialRotaryMulGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(rotary_mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0, 0})
    .OP_END_FACTORY_REG(InplacePartialRotaryMulGrad)
} // namespace ge

#endif
