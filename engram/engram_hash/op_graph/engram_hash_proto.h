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
 * \file engram_hash_proto.h
 * \brief
 */
#ifndef ENGRAM_HASH_H_
#define ENGRAM_HASH_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief EngramHash. \n
* Computes Engram n-gram hash embedding indices. For each layer l, token t and output
* column c, with last = c / num_embed_table_per_ngram + 1 and tb = c %
* num_embed_table_per_ngram: \n
* hash = XOR(int64(ids[t, k]) * mult[l, k], k = 0...last), \n
* hash_indices[l, t, c] = int32(hash mod vocab[l, last - 1, tb]) + offs[l, c]. \n
* The modulo is C-style truncation on int64; a negative remainder is corrected by
* adding the vocab size.

* @par Inputs:
* Four inputs, including:
* @li ngram_token_ids: A Tensor of type int32, with shape (num_tokens, max_ngram_size).
* The n-gram token ids. Format support ND. \n
* @li multipliers: A Tensor of type int64, with shape (num_ngram_layers, max_ngram_size).
* The hash multiplier of each layer and each n-gram position. Format support ND. \n
* @li vocab_sizes: A Tensor of type int32, with shape (num_ngram_layers,
* max_ngram_size - 1, num_embed_table_per_ngram). The vocab size of each embedding
* table. Format support ND. \n
* @li offsets: A Tensor of type int32, with shape (num_ngram_layers, num_out_cols), where
* num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram. The embedding table
* offset (prefix sum of vocab sizes) of each layer and each output column. Format
* support ND. \n

* @par Outputs:
* hash_indices: A Tensor of type int32, with shape (num_ngram_layers, num_tokens,
* num_out_cols). The hash-based embedding indices. Format support ND. \n

* @attention Constraints:
* @li max_ngram_size must be at least 2; num_ngram_layers and num_embed_table_per_ngram
* must be positive.
* @li The shapes of "multipliers", "vocab_sizes", "offsets" and "hash_indices" must
* match the shapes derived above.
* @li num_tokens can be 0, which produces an empty output.
*/
REG_OP(EngramHash)
    .INPUT(ngram_token_ids, TensorType({DT_INT32}))
    .INPUT(multipliers, TensorType({DT_INT64}))
    .INPUT(vocab_sizes, TensorType({DT_INT32}))
    .INPUT(offsets, TensorType({DT_INT32}))
    .OUTPUT(hash_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EngramHash)

} // namespace ge

#endif // ENGRAM_HASH_H_
