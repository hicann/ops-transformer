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
 * \file matmul_reduce_scatter_v2_proto.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_V2_PROTO_H_
#define MATMUL_REDUCE_SCATTER_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion op of matmul and reduce scatter v2.
* @par Inputs:
* six inputs, including:
* @li x1: A matrix tensor. The type supports float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1. The
format supports ND. The x1 only supports two dimensions in current version, for example (M, K). The x1 doesn't support
transposed.
* @li x2: A matrix tensor. The type supports float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1. The
x2 only supports two dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed.
The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range
[256, 65535).
* @li bias: A matrix tensor. The format supports ND. If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, float4_e2m1,
then the bias type supports float32. \n If x1 type is float16, bfloat16, then the bias type supports float16, bfloat16
and the current version doesn't supports the scenario where bias is not 0. \n The bias only supports one dimension in
current version, for expample (N,). \n The bias only supports nullptr in the per_block scenario. \n
* @li x1_scale: A matrix tensor. The type supports float32, float_e8m0. The format supports ND. \n The x1_scale only
supports nullptr when x1's type is float16 or bfloat16. \n The x1_scale only supports one dimension and only one element
in the per_tensor scenario, for expample (1,). \n The x1_scale only supports two dimensions in the per_block scenario,
for expample (ceildiv(M, 128), ceildiv(K, 128)). \n The x1_scale only supports three dimensions in the mxfp scenario,
for expample (M, ceildiv(M, 64), 2). \n
* @li x2_scale: A matrix tensor. The type supports float32, float_e8m0. The format supports ND. \n The x2_scale only
supports nullptr when x1's type is float16 or bfloat16. \n The x2_scale only supports one dimension and only one element
in the per_tensor scenario, for expample (1,). \n The x2_scale only supports two dimensions in the per_block scenario,
for expample (ceildiv(K, 128), ceildiv(N, 128)). \n The x2_scale only supports three dimensions in the mxfp scenario,
for expample (M, ceildiv(M, 64), 2). \n
* @li quant_scale: A matrix tensor. The type supports float32. The format supports ND. The quant_scale only supports one
dimension and only one element, for example (1,). The quant_scale only supports nullptr in current version. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to perform. Default: "sum". The reduce_op only
supports sum in current version.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to [M, K] before multiplication.
* If False, don't change the shape of "x1" before multiplication. Default: "false". The is_trans_a only supports false
in current version.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to [K, N] before multiplication.
* If False, don't change the shape of "x2" before multiplication. Default: "false".
* @li comm_turn: An int. Number of communications with CCU. Default: "0". The comm_turn only supports 0 in current
version.
* @li rank_size: An int. Number of rank num. Default: "0".The rank_size only supports 2/4/8/16/32/64 in current version.
* @li block_size: An int. Number of values in the M, N directions in y correspond to one dequantization coefficient.
Default: "0". The block_size only supports 0 in current version.
* @li group_size: An int. Number of values in the M, N, K directions in x1/x2 correspond to one dequantization
coefficient. \n Default: "0". Reserved attribute. The group_size only supports 549764202624 in the per_block scenario.
The group_size only supports 4295032864 in the mxfp scenario. The group_size only supports 0 in the other scenario. \n
* @li is_amax_out: A bool. whether the output of amax_out is supported . Default: "false". If True, supports output
amax_out.
* @li y_dtype: An int. Default: "0". The y_dtype only supports 0(float32)/1(float16)/27(bfloat16) in current version. \n
* @li comm_mode: A string. communication Mode. Default: "ai_cpu". The comm_mode only supports "ai_cpu" or "aiv" in
current version. \n
*
* @par Outputs:
* @li y: A matrix tensor. If x1 type is float16, bfloat16, The type is same as x1.
  If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, float4_e2m1, the output type supports float16, bfloat16, float32.
The format supports ND. The y is two dimensions, for example (M/rank_size, N).
* @li amax_out: A matrix tensor. The type supports float32. The format supports ND. The amax_out is 1 dimension and only
one element, for example (1,). The amax_out only supports nullptr in current version. \n
*/
REG_OP(MatmulReduceScatterV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(amax_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(MatmulReduceScatterV2)
} // namespace ge
#endif // MATMUL_REDUCE_SCATTER_V2_PROTO_H_
