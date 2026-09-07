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
 * \file batch_mat_mul_reduce_scatter_allto_all_proto.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_PROTO_H_
#define BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion op of batch matmul, reduce scatter, and alltoall.
* @par Inputs:
* Three inputs, including (Tp is short for tp_world_size, and ep is short for ep_world_size.):
* @li x: A matrix Tensor. The type support float16, bfloat16. The x only supports 3 dimensions. The data format can be
ND.
* x (E/ep, ep*C, M/tp).
* @li weight: A matrix Tensor. The type support float16, bfloat16 and its type must be the same as the tpye of x. The
weight only supports 3 dimensions. The data format can be ND.
* weigh (E/ep, M/tp, H).
* @li bias: A matrix Tensor. The type support float16, float32. If x is float16, bias must be float16. If x is bfloat16,
bias must be float32. 2D and 3D are supported. The data format can be ND. \n
* y_shard_type 0: bias (E/ep, 1, H/tp), y_shard_type 1: bias (E/ep, 1, H), 3 dims; \n
* y_shard_type 0: bias (E/ep, H/tp), y_shard_type 1: bias (E/ep, H), 2 dims.
* @par Attributes:
* @li group_ep: A required String identifying the expert group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li group_tp: A required String identifying the tensor group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li ep_world_size: A required int identifying the number of expert parallel group rank num. The value can be 2, 4, 8,
16 or 32.
* @li tp_world_size: A required int identifying the number of tensor parallel group rank num. The value can be 2, 4, 8,
16 or 32.
* @li y_shard_type: An int. Represents the output y shards on dim 2 or 3 in the tensor parallel group.
* Default: "0". The value 0 indicates that ReduceScatter is performed by tensor parallel group in the H dimension (2nd
dimension of the BatchMatMul computation result, which has three dimensions: 0th, 1st, and 2nd).
* The value 1 indicates that ReduceScatter is performed by tensor parallel group in the C dimension (1st dimension of
the BatchMatMul computation result).
* @li transpose_weight: A bool. If True, changes the shape of "weight" from [E, N, K] to
* [E, K, N] before multiplication. Default: "false". \n
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. 3D is supported. The type is the same as that of input x. The
data format can be ND.
* y_shard_type 0: y (E, C, H/tp), y_shard_type 1: y (E, C/tp, H).
*
* @attention Constraints:
* @li The x[0] means E/tp value, y[0] means E, so the x[0] multi tp should equal the y[0].  Other divisible
relationships are similar to this one.
* @li The value range of E is [2, 512], and E is an integer multiple of ep.
* @li he value range of H is [1, 65535], and H is an integer multiple of tp when y_shard_type is 0.
* @li The value range of M/tp is [1, 65535].
* @li The value range of E/ep is [1, 32].
* @li C is greater than 0 and cannot exceed the upper limit of the operator's device memory. C is an integer multiple of
tp when y_shard_type is 1.
* @li Group_ep and group_tp can't be consistent. Ep and tp can only be 2, 4, 8, 16 or 32. Supernodes cannot be crossed.
\n
*/
REG_OP(BatchMatMulReduceScatterAlltoAll)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(group_tp, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(tp_world_size, Int)
    .ATTR(y_shard_type, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulReduceScatterAlltoAll)

} // namespace ge

#endif // BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_PROTO_H_
