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
 * \file swin_transformer_ln_qkv_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Five inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li head_num: A optional attribute, the type is int.
* @li head_dim: A optional attribute, the type is int.
* @li seq_length: A optional attribute, the type is int.
* @li shifts: A optional attribute, the type is list int. Defaults to ().
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-7. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(SwinTransformerLnQKV)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .ATTR(shifts, ListInt, {})
    .ATTR(epsilon, Float, 0.0000001f)
    .OP_END_FACTORY_REG(SwinTransformerLnQKV)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
