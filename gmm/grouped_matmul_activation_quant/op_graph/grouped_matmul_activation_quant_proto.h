/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_activation_quant_proto.h
 * \brief
 */
#ifndef OPS_TRANSFORMER_GROUPED_MATMUL_ACTIVATION_QUANT_PROTO_H_
#define OPS_TRANSFORMER_GROUPED_MATMUL_ACTIVATION_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of grouped matmul, activation and quantization. Current scenario supports
* gelu_tanh activation with MXFP8 and MXFP4 quantization.

* @par Inputs:
* @li x: A tensor. The left matrix of grouped matmul. The shape supports (m, k), the format supports ND,
* and the data type supports float8_e4m3fn, float8_e5m2, float4_e2m1 and float4_e1m2.
* @li group_list: A tensor. Indicates the group information of grouped matmul. The shape supports (e),
* the format supports ND, and the data type supports int64.
* @li weight: A dynamic tensor list. The right matrix of grouped matmul. The data type supports
* float8_e4m3fn, float4_e2m1 and float4_e1m2, and the format supports FRACTAL_NZ. The view shape supports (e, k, n) when
* transpose_weight is false and (e, n, k) when transpose_weight is true. For FLOAT8, the storage shape supports
* (e, ceildiv(n, 32), ceildiv(k, 16), 16, 32) when transpose_weight is false and
* (e, ceildiv(k, 32), ceildiv(n, 16), 16, 32) when transpose_weight is true. For FLOAT4, the last storage
* dimension is 64, namely (e, ceildiv(n, 64), ceildiv(k, 16), 16, 64) and
* (e, ceildiv(k, 64), ceildiv(n, 16), 16, 64), respectively.
* @li weight_scale: A dynamic tensor list. The MX scale of weight. The format supports ND,
* and the data type supports float8_e8m0. The shape supports (e, ceildiv(k, 64), n, 2) when
* transpose_weight is false and (e, n, ceildiv(k, 64), 2) when transpose_weight is true.
* @li bias: A dynamic tensor list. The bias of grouped matmul. The data type supports float32. In the MX
* scenario, bias must be empty.
* @li x_scale: An optional tensor. The MX scale of x. In the MX scenario, this input is required. The shape
* supports (m, ceildiv(k, 64), 2), the format supports ND, and the data type supports float8_e8m0.

* @par Outputs:
* @li y: A tensor. The quantized output after grouped matmul and activation. The logical shape supports (m, n),
* the format supports ND, and the data type supports float8_e4m3fn, float8_e5m2, float4_e2m1 and float4_e1m2.
* In the Torch raw carrier, FLOAT4 is packed as uint8 with physical shape (m, ceildiv(n, 2)); this does not
* change the ACL/GE logical shape.
* @li y_scale: A tensor. The MX scale of y. The shape supports
* (m, ceildiv(n, 64), 2), the format supports ND, and the data type supports float8_e8m0.

* @par Attributes:
* @li activation_type: A string. The activation type. Current scenario only supports "gelu_tanh".
* @li transpose_weight: A bool. Indicates whether weight is transposed. Default: false.
* @li group_list_type: An int. Indicates the group list type. Default: 0.
* @li tuning_config: A list int. Reserved parameter. Default: {0}.
* @li quant_mode: A string. The quantization mode. Current scenario only supports "mx". Default: "".
* @li y_dtype: An int. Indicates the data type of y. It supports DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2,
* DT_FLOAT4_E2M1 and DT_FLOAT4_E1M2. DT_UNDEFINED makes y follow x. Default: DT_UNDEFINED.
* @li round_mode: A string. The rounding mode. Current scenario only supports "rint". Default: "rint".
* @li scale_alg: An int. The scale algorithm. 0 means OCP implementation, 1 means cuBLAS implementation for
* FLOAT8 output, and 2 means dynamic dtype range implementation for FLOAT4_E2M1 output. FLOAT4_E1M2 output only
* supports 0. Default: 0.
* @li dst_type_max: A float. Indicates Amax(DType). Default: 0.0.
*/
REG_OP(GroupedMatmulActivationQuant)
    .INPUT(x, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .DYNAMIC_INPUT(weight_scale, TensorType({DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(activation_type, String)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(group_list_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .ATTR(quant_mode, String, "")
    .ATTR(y_dtype, Int, DT_UNDEFINED)
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0f)
    .OP_END_FACTORY_REG(GroupedMatmulActivationQuant)

} // namespace ge

#endif
