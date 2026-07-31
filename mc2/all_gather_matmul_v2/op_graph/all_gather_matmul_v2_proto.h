/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file all_gather_matmul_v2_proto.h
 * \brief
 */
#ifndef ALL_GATHER_MATMUL_V2_PROTO_H_
#define ALL_GATHER_MATMUL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/* *
* @brief Fusion op of allgather and matmul v2.
* @par Inputs:
* six inputs, including:
* @li x1: A matrix tensor. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, float4_e2m1.
The format support ND. The x1 only support 2 dimensions in current version, for example (M, K). The x1 doesn't
support transposed.
* @li x2: A matrix tensor. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, float4_e2m1.
The format support ND. The x2 only support 2 dimensions in current version, for example (K, N). The x2 support
transposed and non-transposed. The K value in x2 should be same as the K value in x1 when x2 is non-transposed,
and the K value should be in range [256, 65535).
* @li bias: A matrix tensor. If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, then the bias type support float32.
  If x1 type is float16, bfloat16, then the bias type support float16, bfloat16. The format support ND. The current
version not support the scenario where bias is not 0.
* @li x1_scale: A matrix tensor. The type support float32, float_e8m0. The format support ND. The x1_scale only support
1 dimension and only one element in current version, for example (1,).
* @li x2_scale: A matrix tensor. The type support float32, float_e8m0. The format support ND. The x2_scale only support
1 dimension and only one element in current version, for example (1,).
* @li quant_scale: A matrix tensor. The type support float32. The format support ND. The quant_scale only support 1
dimension and only one element, for example (1,). The quant_scale only support nullptr in current version. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
  [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
  [K, N] before multiplication. Default: "false".
* @li gather_index: An int. Represents the input index for doing gather.
  Default: "0". The gather_index only support 0 in current version.
* @li comm_turn: An int. Number of communications with CCU. Default: "0". The comm_turn only support 0 in current
version.
* @li rank_size: An int. Number of rank num. Default: "0".
* @li block_size: An int. Number of values in the M, N directions in y correspond to one dequantization coefficient.
Default: "0". The block_size only supports 0 in current version.
* @li group_size: An int. Number of values in the M, N, K directions in x1/x2 correspond to one dequantization
coefficient. \n Default: "0". Reserved attribute. The block_size only supports 549764202624 in the per_block scenario.
The block_size only supports 0 in the other scenario. \n
* @li is_gather_out: A bool. If True, output gather_out matrix. Default: "true".
* @li is_amax_out: A bool. If True, output amax matrix. Default: "false".
* @li y_dtype: An int. Type of the matmul output. Default: "0". \n
*
* @par Outputs:
* @li y: A matrix tensor. The type support float16, bfloat16, float32. The format support ND. The y is 2 dimension, for
example (M*rank_size, N).
* @li gather_out: A matrix tensor. All data returned from communication. The type of gather_out is consistent with that
of x1. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, float4_e2m1. The format support ND.
* @li amax_out: A matrix tensor. Maximum value of output matrix. The type support float32. The format support ND. The
amax_out is 1 dimension and only one element, for example (1,). The amax_out only support nullptr in current version. \n
* @li comm_mode: A string. Communication mode. Default: "ai_cpu". The comm_mode only supports "ccu" , "ai_cpu" or "aiv"
in current version. \n
*/
REG_OP(AllGatherMatmulV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(gather_out,
    TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .OUTPUT(amax_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(AllGatherMatmulV2)
} // namespace ge


#endif // ALL_GATHER_MATMUL_V2_PROTO_H_
