/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file partial_rotary_mul_quant_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_
#define OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_

#include "graph/operator_reg.h"
#include "common/include/op_host/op_const_def_transformer.h"

namespace ge {
/**
 * @brief Partial rotary position embedding with quantization.
 * @par Inputs:
 * @li x: A tensor. Must be one of the following types: float16, float32, bfloat16.
 * @li cos: A tensor for rotary position embedding cos. Must be one of the following types: float16, float32, bfloat16.
 * @li sin: A tensor for rotary position embedding sin. Must be one of the following types: float16, float32, bfloat16.
 * @li scale: Optional. HiFloat8 multiplier: single-element float32 tensor; absent means 1.0.
 * @par Attributes:
 * @li y_dtype: Optional. 0: HiFloat8 (Ascend950), 1: HiFloat4 (Ascend960DT/Ascend960PR). Defaults to 0.
 * @li quant_mode: Optional. Quantization mode. Defaults to 0.
 * @li round_mode: Optional. HiFloat8 supports "round" and "hybrid" (CAST_HYBRID). Defaults to "round".
 * @li axis: Optional. Axis for quantization. Ignored for HiFloat8. Defaults to -1.
 * @li rotary_mode: Optional. Rotary mode. 0-"half", 1-"interleave", 2-"quarter", 3-"deepseek_interleave". Defaults to
 * 0.
 * @li partial_slice: Optional. Slice range [start, end) for partial rotation. Defaults to [0, 0].
 * For HiFloat8, an omitted or empty slice skips RoPE and still scales and quantizes all of x.
 * @par Outputs:
 * @li y: A tensor of type hifloat8, with the same shape as x.
 * @li y_scale: An empty float32 tensor with shape [0].
 * @attention Constraints:
 * Let (B, S, N, D) represent the shape of x, partial_slice = [start, end], and L = end - start.
 * @li x, cos, sin and y must be 4D tensors in ND format. All dimensions of x must be positive.
 * @li x, cos and sin must have the same data type, and cos and sin must have identical shapes.
 * @li D must satisfy 0 < D <= 1024.
 * @li A non-empty partial_slice must contain two integers satisfying 0 <= start <= end <= D.
 * When L > 0, end must equal D.
 * @li When L > 0, the last dimension of cos and sin must equal L.
 * @li Ascend 950PR&950DT products:
 *  - y_dtype must be 0 (HiFloat8). rotary_mode supports 0 (half), 1 (interleave), 2 (quarter) and
 *    3 (deepseek_interleave).
 *  - When L > 0, both D and L must be multiples of 2 for rotary_mode 0, 1 or 3, and multiples of 4 for
 *    rotary_mode 2.
 *  - When L > 0, each of the first three dimensions of cos and sin must equal the corresponding dimension of x or 1.
 *  - When start == end, RoPE is skipped and the entire x is scaled and quantized. The rotary-mode divisibility
 *    and cos/sin broadcasting and last-dimension constraints do not apply.
 *  - quant_mode and axis are ignored; rotation and quantization use the last dimension.
 */
REG_OP(PartialRotaryMulQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(y_dtype, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(round_mode, String, "round")
    .ATTR(axis, Int, -1)
    .ATTR(rotary_mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0, 0})
    .OUTPUT(y, TensorType({DT_HIFLOAT8, Ops::Transformer::DT_HIFLOAT4}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT, Ops::Transformer::DT_HIFLOAT4_SCALE}))
    .OP_END_FACTORY_REG(PartialRotaryMulQuant)

} // namespace ge

#endif
