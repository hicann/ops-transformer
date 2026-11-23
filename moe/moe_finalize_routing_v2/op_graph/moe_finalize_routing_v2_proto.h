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
 * \file moe_finalize_routing_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2_H_
#define OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2_H_

#include "graph/operator_reg.h"
#include "common/include/op_host/op_const_def_transformer.h"

namespace ge {

/**
* @brief In MoE computation, processes and merges the MoE FFN outputs into the original token sequence.
* @par Inputs:
* @li expanded_x: MoE FFN output. A 2D or 3D Tensor of type BFloat16, Float16, Float32 or Float4E2M1. Format: ND.
Shape is (NUM\_ROWS \* K, H) for dropless scenarios or (E, C, H) for drop/pad scenarios, where E is the number of
experts, C is the expert capacity, and H is the logical hidden size. For Float4E2M1 inputs, dropless shape is
(R, H), where R is the physical row count; drop/pad shape is (E, C, H), where R = E * C. H must be positive and
even, and expanded_x must be contiguous.
* @li expanded_row_idx: Token index mapping. A 1D Int32 Tensor with shape (NUM\_ROWS \* K). Values are in
[0, NUM\_ROWS \* K - 1] for drop_pad_mode 0/2 or [-1, E \* C - 1] for drop_pad_mode 1/3.
For Float4E2M1 inputs, data and expanded_x_scale use the same physical row index; indexes outside [0, R) skip the
entire route, including its bias.
* @li x1: First shared-expert output. An optional 2D Tensor of type BFloat16, Float16 or Float32. Shape matches y.
Type matches expanded_x, except that Float4E2M1 inputs require BFloat16.
* @li x2: Second shared-expert output. An optional 2D Tensor of type BFloat16, Float16 or Float32. Shape matches y.
Type matches expanded_x, except that Float4E2M1 inputs require BFloat16. Requires x1 to be present.
* @li bias: Expert bias. An optional 2D Tensor of type BFloat16, Float16 or Float32 with shape (E, H).
Type matches expanded_x, except that Float4E2M1 inputs require BFloat16. Requires expert_idx to be present.
* @li scales: Routing weights. An optional Tensor of type BFloat16, Float16 or Float32 with shape (NUM\_ROWS, K).
Type matches expanded_x except on Ascend 950PR&950DT products. Float4E2M1 inputs use Float32 routing weights.
For HIFLOAT4 inputs, this tensor supplies the Float32 dequantization scales with shape (NUM\_ROWS, K, H / 64).
* @li expert_idx: Expert indexes for bias. An optional 2D Int32 Tensor with shape (NUM\_ROWS, K).
Values must be in [0, E - 1]. Required when bias is present.
* @li x: Input for copy or constant experts. An optional 2D Tensor of type BFloat16, Float16 or Float32.
Type matches expanded_x. Shape matches y. Must be omitted for Float4E2M1 inputs.
* @li alpha1: First scaling coefficient for constant experts. An optional 2D Tensor of type BFloat16, Float16 or
Float32. Type matches expanded_x. Shape is (constant_expert_end - constant_expert_start, H).
Must be omitted for Float4E2M1 inputs.
* @li alpha2: Second scaling coefficient for constant experts. An optional 2D Tensor of type BFloat16, Float16 or
Float32. Type matches expanded_x. Shape is (constant_expert_end - constant_expert_start, H).
Must be omitted for Float4E2M1 inputs.
* @li v: Offset for constant experts. An optional 2D Tensor of type BFloat16, Float16 or Float32.
Type matches expanded_x. Shape is (constant_expert_end - constant_expert_start, H).
Must be omitted for Float4E2M1 inputs.
* @li expanded_x_scale: An optional Float8E8M0 Tensor in ND format. Required for Float4E2M1.
Ignored for other input types.
The tensor must be contiguous and has shape (R, ceil(H / 64), 2). Each scale dequantizes 32 elements from the
corresponding physical row.
* @par Outputs:
* @li y: Merged token sequence. A 2D Tensor of type BFloat16, Float16 or Float32 with shape (NUM\_ROWS, H).
Type matches expanded_x, except that Float4E2M1 inputs produce BFloat16. For these inputs, dequantized data,
optional bias and shared-expert outputs accumulate in Float32 and are converted once to BFloat16.
Float4E2M1 inputs do not support zero, copy, or constant experts.
* @par Attributes:
* @li drop_pad_mode: Drop/pad behavior and index arrangement. Type is Int. Default: 0. Range: [0, 3].
0: dropless, column arrangement; 1: drop/pad, column arrangement;
2: dropless, row arrangement; 3: drop/pad, row arrangement.
* @li zero_expert_range: Optional ListInt [zero_expert_start, zero_expert_end]. Default: {}.
zero_expert_start must be greater than or equal to 0, zero_expert_end must be less than or equal to 10240,
and zero_expert_start must be less than zero_expert_end. Must be empty for Float4E2M1 inputs.
* @li copy_expert_range: Optional ListInt [copy_expert_start, copy_expert_end]. Default: {}.
copy_expert_start must be greater than or equal to 0, copy_expert_end must be less than or equal to 10240,
and copy_expert_start must be less than copy_expert_end. Must be empty for Float4E2M1 inputs.
* @li constant_expert_range: Optional ListInt [constant_expert_start, constant_expert_end]. Default: {}.
constant_expert_start must be greater than or equal to 0, constant_expert_end must be less than or equal to 10240,
and constant_expert_start must be less than constant_expert_end. Must be empty for Float4E2M1 inputs.
* @li k: Number of top-K experts selected per token. Optional Int. Default: 1.
Determines K when scales is absent. For Float4E2M1 inputs, k must be positive, the index count must be divisible
by k, and NUM\_ROWS is the index count divided by k; k is independent of the scale shape.
*/
REG_OP(MoeFinalizeRoutingV2)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, Ops::Transformer::DT_HIFLOAT4, DT_FLOAT4_E2M1}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(expert_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(alpha1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(alpha2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(expanded_x_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(zero_expert_range, ListInt, {})
    .ATTR(copy_expert_range, ListInt, {})
    .ATTR(constant_expert_range, ListInt, {})
    .ATTR(k, Int, 1)
    .OP_END_FACTORY_REG(MoeFinalizeRoutingV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2_H_
