/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_PROTO_H_
#define SPARSE_FLASH_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Function SparseFlashAttention.
 *
 * @par Inputs:
 * @li query: A matrix tensor. The type support float16, bfloat16.
 * Query for attention structure.
 * @li key: A matrix tensor. The type support float16, bfloat16.
 * Key for attention structure.
 * @li value: A matrix tensor. The type support float16, bfloat16.
 * Value for attention structure.
 * @li sparse_indices: A matrix tensor. The type support int32.
 * The indices of sparse kv cache.
 * @li block_table: A matrix tensor. The type support int32.
 * The block mapping table used in KV storage of PageAttention.
 * @li actual_seq_lengths_query: A matrix tensor. The type support int32.
 * Efective sequence length of query in different batches.
 * @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
 * Effective sequence length of key and value in different batches.
 * @li query_rope: A tensor. The type support float16, bfloat16.
 * @li key_rope: A tensor. The type support float16, bfloat16.
 * @li sinks: A tensor. The type support float.
 *
 * @par Attributes:
 * @li scale_value: A float. A required attribute.
 * @li sparse_block_size: An int. An optional attribute. Max value: 64. Default: 1.
 * @li layout_query: A string. An optional attribute. Specifies the layout of `query`, the value must be one of
 * ["BSND", "TND"]. Default: "BSND".
 * @li layout_kv: A string. An optional attribute. Specifies the layout of `key` and 'value', the value must be one of
 * ["BSND", "TND", "PA_BSND"]. Default: "BSND".
 * @li sparse_mode: An int. Sparse mode. Default: 3.
 * - 0: default mask
 * - 3: rightDownCausal make
 * - 4: band mask
 * @li pre_tokens: An int. Previous tokens. Default: 9223372036854775807.
 * @li next_tokens: An int. Next tokens. Default: 9223372036854775807.
 * @li attention_mode: An int. An optional attribute. Default: 0.
 * @li return_softmax_lse: A bool. An optional attribute. Default: false.
 *
 * @par Outputs:
 * @li attention_out: A matrix tensor. The type support float16, bfloat16.
 * @li softmax_max: A matrix tensor. The type support float32.
 * @li softmax_sum: A matrix tensor. The type support float32.
 */
// SparseFlashAttention is also declared in the built-in ops_proto_legacy.h of the CANN package; the shared
// OPS_PROTO_DEF guard below keeps the two declarations from being compiled twice in one translation unit.
#ifndef OPS_PROTO_DEF_SPARSEFLASHATTENTION
#define OPS_PROTO_DEF_SPARSEFLASHATTENTION
REG_OP(SparseFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(sinks, TensorType({DT_FP32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(softmax_max, TensorType({DT_FP32}))
    .OUTPUT(softmax_sum, TensorType({DT_FP32}))
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 0)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttention)
#endif

} // namespace ge
#endif // SPARSE_FLASH_ATTENTION_PROTO_H_
