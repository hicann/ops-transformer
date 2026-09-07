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
 * \file grouped_mat_mul_allto_allv_proto.h
 * \brief
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_
#define GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion of grouped matmul and alltoallv.

* @par Inputs:
* @li gmm_x: A matrix tensor of shape [A, H1]. The data type of elements supports float16, bfloat16, hifloat8,
      float8_e5m2, float8_e4m3fn or float4_e2m1; the format supports ND.
* @li gmm_weight: A matrix tensor of shape [e, H1, N1]. The data type of elements supports float16, bfloat16, hifloat8,
      float8_e5m2, float8_e4m3fn or float4_e2m1; the format supports ND.
* @li gmm_x_scale: A matrix Tensor. The type support float32 or float8_e8m0. The format supports ND.
* @li gmm_weight_scale: A matrix Tensor. The type support float32 or float8_e8m0. The format supports ND.
* @li send_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format
supports ND.
* @li recv_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format
supports ND.
* Data type of elements is int64. Format: ND.
* @li mm_x: A matrix tensor of shape [BS, H2]. The data type of elements supports
      float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn or float4_e2m1; the format supports ND.
* @li mm_weight: gmm_weight: A matrix tensor of shape [H2, N2]. The data type of elements supports
      float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn or float4_e2m1; the format supports ND.
* @li mm_x_scale: A matrix Tensor. The type support float32 or float8_e8m0. The format supports ND.
* @li mm_weight_scale: A matrix Tensor. The type support float32 or float8_e8m0. The format supports ND.
* @li comm_quant_scale: A matrix Tensor. The type support float32. The format supports ND.

* @par Attributes:
* @li group: A required String identifying the expert group of ranks
* @li ep_world_size: A required int identifying the number of expert parallel group rank num.
* @li send_counts: An int list. A list containing amount of data to be sent.
* @li recv_counts: An int list. A list containing amount of data to be received.
* @li gmm_x_quant_mode: An int. Quantization mode of gmm_x. Default: 0.
*        - 0：No Quantization
*        - 1：PerTensor Quantization
*        - 2：PerChannel Quantization
*        - 3：PerToken Quantization
*        - 4：PerGroup Quantization
*        - 5：PerBlock Quantization
*        - 6：Mx Quant Quantization
* @li gmm_weight_quant_mode: An int. Quantization mode of gmm_weight. Default: 0.
* @li trans_gmm_weight: A boolean value. Indicating whether gmm_weight is transposed.
* @li trans_mm_weight: A boolean value. Indicating whether mm_weight is transposed.
* @li mm_x_quant_mode: An int. Quantization mode of mm_x. Default: 0.
* @li mm_weight_quant_mode: An int. Quantization mode of mm_weight. Default: 0.
* @li comm_quant_mode: An int. Quantization mode of communication. Default: 0.
* @li group_size: An int. Default: 0.
* @li y_dtype: An int. Default: DT_UNDEFINED
* @li mm_dtype: An int. Default: DT_UNDEFINED
* @li comm_quant_dtype: An int. Default: 0.
* @li comm_mode: A String value. Indicating communication type. Default: ai_cpu.

* @par Outputs:
* @li y: A matrix tensor of shape [BSK, N1] containing result of grouped matmul. The data type of elements supports
float16 or bfloat16; the format supports ND.
* @li mm_y_optional: A matrix tensor of shape [BS, N2] containing result of matmul. The data type of elements supports
float16 or bfloat16; the format supports ND.
*/

REG_OP(QuantGroupedMatMulAlltoAllv)
    .INPUT(gmm_x, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .INPUT(gmm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .INPUT(gmm_x_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .INPUT(gmm_weight_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(send_counts_tensor, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(recv_counts_tensor, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(mm_x,
                    TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(mm_weight,
                    TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(mm_x_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(mm_weight_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(comm_quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(mm_y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(send_counts, ListInt)
    .REQUIRED_ATTR(recv_counts, ListInt)
    .REQUIRED_ATTR(gmm_x_quant_mode, Int)
    .REQUIRED_ATTR(gmm_weight_quant_mode, Int)
    .ATTR(trans_gmm_weight, Bool, false)
    .ATTR(trans_mm_weight, Bool, false)
    .ATTR(mm_x_quant_mode, Int, 0)
    .ATTR(mm_weight_quant_mode, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(y_dtype, Int, DT_UNDEFINED)
    .ATTR(mm_dtype, Int, DT_UNDEFINED)
    .ATTR(comm_quant_dtype, Int, 0)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(QuantGroupedMatMulAlltoAllv)

} // namespace ge

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_
