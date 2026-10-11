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
 * \file moe_finalize_routing_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEFINALIZE_ROUTING_H_
#define OPS_OP_PROTO_INC_MOEFINALIZE_ROUTING_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Shape support(NUM\_ROWS \* K, H).
* @li x1: A 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.
* @li x2: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out. If the parameter A is not entered,
      the parameter B can also not be entered.
* @li bias: A 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX.Shape support(E, H). E is the total number of experts, and H is the number of columns.
* @li scales: A 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX.Shape support(NUM\_ROWS, K).
* @li expanded_row_idx: A 1D Tensor. Type is:Int32.Shape support(NUM\_ROWS \* K).Values in Tensor are
      [0,NUM\_ROWS \* K-1].
* @li expanded_expert_idx: A 2D Tensor. Type is Int32. Shape support(NUM\_ROWS, K).
      Values in Tensor are [0, E-1].
* @par Outputs:
* @li y: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Shape support(NUM\_ROWS, H).
*/
REG_OP(MoeFinalizeRouting)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32}))
    .INPUT(expanded_expert_idx, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeFinalizeRouting)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEFINALIZE_ROUTING_H_
