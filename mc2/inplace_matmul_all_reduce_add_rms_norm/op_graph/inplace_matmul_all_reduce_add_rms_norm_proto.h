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
 * \file inplace_matmul_all_reduce_add_rms_norm_proto.h
 * \brief
 */
#ifndef INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_PROTO_H_
#define INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Function InplaceMatmulAllReduceAddRmsNorm.
* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li gamma: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks
*  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
*  perform. support "sum", "min", "max" ,"prod", currently only support "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
*  [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
*  [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: An int. Number of per-group for quant. Default: 0.
* @li epsilon: A float32. Default: 1e-06. \n

* @par Outputs:
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li norm_out: A matrix Tensor. The type support float16, bf16. \n

* @attention Constraints:
* - Constraints for InplaceMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that [MatmulAllReduce].
*  InplaceMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator, and
*  their k axes are equal. If bias is not empty, it is one-dimensional and its dimension is (n).
* @li The input residual must be 3-dimensional and its dimension are (b, s, n). When x1 is two-dimensional,
*  (b*s) of residual is equal to s of x1. The input gamma must be one-dimensional and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data types of x1, x2, bias (if supported), residual, gamma, y,
*  and normOut computation input must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 products support
1, 2, 4, and 8 cards.
*
* - Constraints for InplaceWeightQuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  InplaceWeightQuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator.
*  Their k axes are equal. The range of k and n is [1, 65535]. If bias is not empty,
*  bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional,
* (b*s) of residual is equal to s of x1. gamma must be one-dimensional and its dimension is (n).
* @li The shape of antiquantScale is [1] in the per-tensor scenario; [1,n]\[n] in the per-channel scenario;
*  and [ceil(k,antiquantGroupSize),n] in the per-group scenario. If antiquantOffset is not empty,
*  the shape is the same as that of antiquantScale.
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data type of x2 must be int8 or int4. The data types of x1, bias (if supported), residual,
*  gamma, y, or normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of antiquantGroupSize falls in the range of [32, min(k-1, INT_MAX)] and must be a multiple of 32.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 products support
1, 2, 4, and 8 cards.
*
* - Constraints for InplaceQuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  InplaceQuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li The input x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must
*  be 2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator,
*  and their k axes are equal. If bias is not empty, bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional, (b*s) of
*  residual is equal to s of x1. The input gamma must be one-dimensional and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual. If
*  bias is not empty, the shape size is the same as the last dimension.
* @li If the output residual is of FLOAT16 type, the type of dequantScale is UINT64. If the output residual
*  is of BFLOAT16 type, the type of dequantScale is BFLOAT16.
* @li When the data types of x1 and x2 are INT8, and the data type of bias (if supported) is INT32, the
*  input data types of residual, gamma, y, and normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 products support
1, 2, 4, and 8 cards.
*/
REG_OP(InplaceMatmulAllReduceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64}))
    .OUTPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(epsilon, Float, 1e-06f)
    .OP_END_FACTORY_REG(InplaceMatmulAllReduceAddRmsNorm)

} // namespace ge

#endif // INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_PROTO_H_
