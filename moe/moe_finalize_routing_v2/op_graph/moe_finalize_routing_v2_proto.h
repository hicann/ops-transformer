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

namespace ge {

/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: expandedX in the formula, represents the token sequences. A 2D or 3D Tensor. Type is:BFloat16, Float16
or Float32. Format support ND. Dropless scenario shape is (NUM\_ROWS \* K, H), dropPad scenario shape is (expert_num,
expert_capacity, H).
* @li expanded_row_idx: A 1D Tensor, represents the token indexes of expanded_x. Type is:Int32. Shape support(NUM\_ROWS
\* K).Values in Tensor are [0, NUM\_ROWS \* K – 1] when drop_pad_mode is 0,2; Values in Tensor are [-1, NUM\_ROWS \* K –
1] when drop_pad_mode is 1, 3.
* @li x1: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.
* @li x2: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.If the parameter A is not entered,
      the parameter B can also not be entered.
* @li bias: An optional 2D Tensor, represents the bias of expanded_x. Type is:BFloat16, Float16 or Float32.The data type
requirement of A is consistent with expandedX.Shape support(E, H). E is the total number of experts, and H is the number
of columns.
 * @li scales: An optional 2D Tensor, represents the scale of expanded_x. Type is:BFloat16, Float16 or Float32.The data
type requirement of A is consistent with expandedX except in Ascend 950PR&950DT products. Shape support(NUM\_ROWS, K),
When scales is null, K is determined by the attribute k (default 1).
* @li expert_idx: An optional 2D Tensor, represents the indexes of bias. Type is Int32.Shape support(NUM\_ROWS,
K).Values in Tensor are [0, E-1], if bias exists, expert_idx must exist.
* @li x: An optional 2D Tensor, needed for copy or constant expert. Type is:BFloat16, Float16 or Float32.The data type
requirement of A is consistent with expandedX,and the shape requirements are consistent with the shape of out.
* @li alpha1: An optional 2D Tensor, needed for constant expert. Type is:BFloat16, Float16 or Float32.The data type
requirement of A is consistent with expandedX,and the shape requirements are consistent with the shape of out.
* @li alpha2: An optional 2D Tensor, needed for constant expert. Type is:BFloat16, Float16 or Float32.The data type
requirement of A is consistent with expandedX,and the shape requirements are consistent with the shape of out.
* @li v: An optional 2D Tensor, needed for constant expert. Type is:BFloat16, Float16 or Float32.The data type
requirement of A is consistent with expandedX,and the shape requirements are consistent with the shape of out.
* @par Outputs:
* @li y: A 2D Tensor. Type is:BFloat16, Float16 or Float32.Shape support(NUM\_ROWS, H).
* @par Attributes:
* @li drop_pad_mode: drop mode. Type is Int32. Default: 0, range [0,3].
      0 (dropless scenario, expanded_row_idx column arrangement), 1 (drop or pad scenario, expanded_row_idx column
arrangement), 2 (dropless scenario, expanded_row_idx line arrangement), 3 (drop or pad scenario, expanded_row_idx line
arrangement).
* @li zero_expert_range:  Optional parameter. Type is:ListInt. Like [zero_expert_start, zero_expert_end].
      zero_expert_start must be greater than or equal to 0, zero_expert_end must be less than or equal to 10240,
      zero_expert_start must be less than zero_expert_end. Default: {}.
* @li copy_expert_range:  Optional parameter. Type is:ListInt. Like [copy_expert_start, copy_expert_end].
      copy_expert_start must be greater than or equal to 0, copy_expert_end must be less than or equal to 10240,
      copy_expert_start must be less than copy_expert_end. Default: {}.
* @li constant_expert_range:  Optional parameter. Type is:ListInt. Like [constant_expert_start, constant_expert_end].
      constant_expert_start must be greater than or equal to 0, constant_expert_end must be less than or equal to 10240,
      constant_expert_start must be less than constant_expert_end. Default: {}.
* @li k:  Optional parameter. Type is Int. Represents the number of top-K experts selected per token. Default: 1.
*/
REG_OP(MoeFinalizeRoutingV2)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
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
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(zero_expert_range, ListInt, {})
    .ATTR(copy_expert_range, ListInt, {})
    .ATTR(constant_expert_range, ListInt, {})
    .ATTR(k, Int, 1)
    .OP_END_FACTORY_REG(MoeFinalizeRoutingV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2_H_
