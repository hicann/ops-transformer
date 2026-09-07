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
 * \file grouped_mat_mul_all_reduce_proto.h
 * \brief
 */
#ifndef GROUPED_MAT_MUL_ALL_REDUCE_PROTO_H_
#define GROUPED_MAT_MUL_ALL_REDUCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Function GroupedMatMulAllReduce. This op computes multi groups of matmuls on multi-cards environment.

* @par Inputs:
* @li x: A Tensor List, contains all left matrixs of inputs for matmuls. For each tensor, the data type of elements
supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.
* 32B-aligned size of each dim should be smaller than 2147483647. The size of inner axis should be smaller than 65536.
* @li weight: A Tensor List of weight, contains all right matrixs of inputs for matmul. For each tensor, the data type
of elements supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.
* 32B-aligned size of each dim should be smaller than 2147483647. The size of inner axis should be smaller than 65536.
* @li bias: A Tensor List of bias, contains all bias of inputs for matmul. For each tensor, the data type of elements
supports float16 or float32; the format supports ND. The maximum length allowed is 64.
* @li group_list: a Tensor, indicates M-axis distributation of groups of matmuls for inputs and outputs.
* Data type of elements is int64. Format: ND. The maximum length allowed is 64.

* @par Attributes:
* @li splitItem: An int64, indicates whether do tensor split for inputs and outputs.
* 0: no split for inputs and outputs; 1: inputs need tensor split; 2: outputs need tensor split;
* 3: both inputs and outputs need tensor split. Default value is 0.
* @li group: A string. A required String identifying the group of ranks.
* @li reduceOp: A string. A required string identifying the reduction operation to
 perform. support "sum".
* @li commTurn: An int64. Number of communications with AICPU. Default: 0.

* @par Outputs:
* y: A Tensor List, contains all result of groups of matmuls. For each tensor,
* the data type of elements supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.

* @attention Constraints:
* Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(GroupedMatMulAllReduce)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(splitItem, Int, 0)
    .REQUIRED_ATTR(group, String)
    .ATTR(reduceOp, String, "sum")
    .ATTR(commTurn, Int, 0)
    .OP_END_FACTORY_REG(GroupedMatMulAllReduce)

} // namespace ge

#endif // GROUPED_MAT_MUL_ALL_REDUCE_PROTO_H_
