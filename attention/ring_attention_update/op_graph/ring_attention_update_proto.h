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
 * \file ring_attention_update_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_RING_ATTENTION_UPDATE_OPS_H_
#define OPS_OP_PROTO_INC_RING_ATTENTION_UPDATE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Update multi output of RingAttention.

* @par Inputs:
* seven inputs, including:
* @li prev_attn_out: A matrix Tensor. The type support float16, bf16, float32.
* @li prev_softmax_max: A matrix Tensor. The type support float32.
* @li prev_softmax_sum: A matrix Tensor. The type support float32.
* @li cur_attn_out: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li cur_softmax_max: A matrix Tensor. An optional input parameter. The type support float32.
* @li cur_softmax_sum: A matrix Tensor. An optional input parameter. The type support float32.
* @li actual_seq_qlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the attn_out seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].

* @par Attributes:
* @li input_layout: A string. A optional attribute. Specifies the layout of `attn_out`,
* the value must be one of ["SBH"]. Default: "SBH".

* @par Outputs:
* @li attn_out: A matrix Tensor. The type support float16, bf16, float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
*/
REG_OP(RingAttentionUpdate)
    .INPUT(prev_attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(prev_softmax_max, TensorType({DT_FLOAT32}))
    .INPUT(prev_softmax_sum, TensorType({DT_FLOAT32}))
    .INPUT(cur_attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(cur_softmax_max, TensorType({DT_FLOAT32}))
    .INPUT(cur_softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OUTPUT(attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .ATTR(input_layout, String, "SBH")
    .ATTR(input_softmax_layout, String, "")
    .OP_END_FACTORY_REG(RingAttentionUpdate)

} // namespace ge

#endif
