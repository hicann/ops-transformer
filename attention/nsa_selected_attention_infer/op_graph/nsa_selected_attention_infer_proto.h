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
 * \file nsa_selected_attention_infer_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_
#define OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Function NsaSelectedAttentionInfer.
 *
 * @par Inputs:
 * @li query: A tensor. The type only support float16, bf16.
 * @li key: A tensor. The type only support float16, bf16.
 * @li value: A tensor. The type only support float16, bf16.
 * @li topk_indices: A tensor. The type only support int32.
 * @li atten_mask: A optional tensor. The type only support bool, uint8.
 * @li block_table: A optional tensor. The type only support int32.
 * @li actual_q_seq_lengths: A optional tensor. The type only support int64.
 * @li actual_kv_seq_lengths: A optional tensor. The type only support int64.
 *
 * @par Attributes:
 * @li input_layout: A required String.
 * @li num_heads: A required Int.
 * @li num_key_value_heads: A required Int.
 * @li selected_block_size: A required Int.
 * @li selected_block_count: A required Int.
 * @li block_size: A required Int.
 * @li scale_value: A required Float.
 * @li sparse_mode: A required Int.
 *
 * @par Outputs:
 * @li attention_out: A tensor. The type only support float16, bf16.
 *
 * @platform: KirinX90, Kirin9030
 *
 */
REG_OP(NsaSelectedAttentionInfer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(topk_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_q_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_kv_seq_lengths, TensorType({DT_INT64}))
    .ATTR(input_layout, String, "BSND")
    .ATTR(num_heads, Int, 1)
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(selected_block_size, Int, 1)
    .ATTR(selected_block_count, Int, 1)
    .ATTR(block_size, Int, 1)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(sparse_mode, Int, 0)
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(NsaSelectedAttentionInfer)

} // namespace ge

#endif // OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_
