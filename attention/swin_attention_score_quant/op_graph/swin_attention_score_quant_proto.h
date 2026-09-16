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
 * \file swin_attention_score_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_SWIN_ATTENTION_SCORE_QUANT_PROTO_H_
#define OPS_QUANT_SWIN_ATTENTION_SCORE_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Quantized window attention score (W8A8): computes attention scores from int8
*        query/key/value with dequant scales, softmax and requantized PV matmul.
*        Currently used by Swin window attention.

* @par Inputs:
* @li query: A tensor of type int8, format ND. Shape [b, n, s, h].
* @li key: A tensor of type int8, format ND.
* @li value: A tensor of type int8, format ND.
* @li scale_quant: A tensor of type float16, format ND. Quantization scale for the
*        softmax output (P) before the PV matmul.
* @li scale_dequant1: A tensor of type uint64, format ND. Dequant scale applied to the
*        quantized key operand of the QK^T matmul, one per key position (s values).
* @li scale_dequant2: A tensor of type uint64, format ND. Dequant scale applied to the
*        quantized value operand of the PV matmul, one per channel of h.
* @li bias_quant: An optional tensor of type float16, format ND. Quantization bias for P.
* @li bias_dequant1: An optional tensor of type int32, format ND. Bias added to the
*        QK^T matmul result, one per key position (s values).
* @li bias_dequant2: An optional tensor of type int32, format ND. Bias added to the
*        PV matmul result, one per channel of h.
* @li padding_mask1: An optional tensor of type float16, format ND. Additive mask applied
*        to the attention score before softmax, shape [n, s, s].
* @li padding_mask2: An optional tensor of type float16, format ND. Currently reserved
*        and not consumed by the operator.

* @par Attributes:
* @li query_transpose: An optional bool. Whether query is transposed, default is false.
* @li key_transpose: An optional bool. Whether key is transposed, default is false.
* @li value_transpose: An optional bool. Whether value is transposed, default is false.
* @li softmax_axes: An optional int. The axis of softmax, default is -1.

* @par Outputs:
* @li attention_score: A tensor of type float16, format ND. Shape [b, n, s, h].
*/
REG_OP(SwinAttentionScoreQuant)
    .INPUT(query, TensorType({DT_INT8}))
    .INPUT(key, TensorType({DT_INT8}))
    .INPUT(value, TensorType({DT_INT8}))
    .INPUT(scale_quant, TensorType({DT_FLOAT16}))
    .INPUT(scale_dequant1, TensorType({DT_UINT64}))
    .INPUT(scale_dequant2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias_quant, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_dequant1, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias_dequant2, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(padding_mask1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask2, TensorType({DT_FLOAT16}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(value_transpose, Bool, false)
    .ATTR(softmax_axes, Int, -1)
    .OP_END_FACTORY_REG(SwinAttentionScoreQuant)
} // namespace ge

#endif // OPS_QUANT_SWIN_ATTENTION_SCORE_QUANT_PROTO_H_
