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
 * \file swin_attention_ffn_proto.h
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
* Four inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16.
* @li x2: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16.
* @li x3: A optional Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li shifts: A optional attribute, the type is list int. Defaults to (). \n

* @par Outputs:
* One output, including:
* @li y: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(SwinAttentionFFN)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(shifts, ListInt, {})
    .OP_END_FACTORY_REG(SwinAttentionFFN)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
