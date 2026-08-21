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
 * \file matmul_allto_all_proto.h
 * \brief 图模式原型定义
 */
#ifndef MATMUL_ALLTO_ALL_PROTO_H
#define MATMUL_ALLTO_ALL_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief Fusion op of alltoall and matmul.
 * @par Inputs:
 * two inputs, including:
 * @li x1: A matrix Tensor. The type support bfloat16, float16, float8_e4m3fn, float8_e5m2, float4_e2m1. The format
 * supports ND.
 * @li x2: A matrix Tensor. The type support bfloat16, float16, float8_e4m3fn, float8_e5m2, float4_e2m1. The format
 * supports ND.
 * @li bias: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @li x1_scale: A matrix Tensor. The type support float, float8_e8m0. The format supports ND.
 * @li x2_scale: A matrix Tensor. The type support float, float8_e8m0. The format supports ND.
 * @li comm_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li x1_offset: A matrix Tensor. The type support float. The format supports ND.
 * @li x2_offset: A matrix Tensor. The type support float. The format supports ND.
 * @par Outputs:
 * @li y: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @par Attributes:
 * @li group: A string. Communication domain identifier.
 * @li world_size: An int.
 * @li all2all_axes: An ListInt. Indicate the data direction for All2All communication. Default: {-1, -2}.
 * @li y_dtype: An int. Declare the output dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li x1_quant_mode: An int. Quantization mode of x1, which support 0(non-quant), 3(pertoken-quant), 6(mx-quant).
 * Default: 0.
 * @li x2_quant_mode: An int. Quantization mode of x2, which support 0(non-quant), 2(perchannel-quant), 6(mx-quant).
 * Default: 0.
 * @li comm_quant_mode: An int. Quantitative types for communication. Default: 0.
 * @li comm_quant_dtype: An int. Communication accuracy. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li transpose_x1: A bool. Whether x1 is transposed. Default: false.
 * @li transpose_x2: A bool. Whether x2 is transposed. Default: false.
 * @li group_size: An int. Default: 0.
 * @li comm_mode: A string. Communication engine. Default: "".
 */
REG_OP(MatmulAlltoAll)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(comm_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .ATTR(all2all_axes, ListInt, {-1, -2})
    .ATTR(y_dtype, Int, 28)
    .ATTR(x1_quant_mode, Int, 0)
    .ATTR(x2_quant_mode, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(comm_quant_dtype, Int, 28)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(MatmulAlltoAll)
} // namespace ge

#endif // MATMUL_ALLTO_ALL_PROTO_H
