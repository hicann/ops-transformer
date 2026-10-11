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
 * \file swin_transformer_ln_qkv_quant_proto.h
 * \brief
 */
#ifndef SWIN_TRANSFORMER_LN_QKV_QUANT_PROTO_H
#define SWIN_TRANSFORMER_LN_QKV_QUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Eight inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: int8.
* @li bias: A Tensor. Must be one of the following types: float16.
* @li quant_scale: A Tensor. Must be one of the following types: float16.
* @li quant_offset: A Tensor. Must be one of the following types: float16.
* @li dequant_scale: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li head_num: A required attribute, the type is int. Defaults to 1.
* @li seq_length: A required attribute, the type is int. Defaults to 32.
* @li epsilon: A required attribute, the type is float. Defaults to 0.000001.
* @li ori_height: A required attribute, the type is int. Defaults to 7
* @li ori_weight: A required attribute, the type is int. Defaults to 7. \n
* @li h_win_szie: A required attribute, the type is int. Defaults to 7. \n
* @li w_win_size: A required attribute, the type is int. Defaults to 7. \n
* @li weight_transpose: A required attribute, the type is bool. Defaults to true. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n
*/
REG_OP(SwinTransformerLnQkvQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT16}))
    .INPUT(quant_offset, TensorType({DT_FLOAT16}))
    .INPUT(dequant_scale, TensorType({DT_UINT64}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .REQUIRED_ATTR(epsilon, Float)
    .REQUIRED_ATTR(ori_height, Int)
    .REQUIRED_ATTR(ori_weight, Int)
    .REQUIRED_ATTR(h_win_szie, Int)
    .REQUIRED_ATTR(w_win_size, Int)
    .REQUIRED_ATTR(weight_transpose, Bool)
    .OP_END_FACTORY_REG(SwinTransformerLnQkvQuant)

} // namespace ge

#endif // SWIN_TRANSFORMER_LN_QKV_QUANT_PROTO_H
