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
 * \file causal_conv1d_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Causal 1D depthwise convolution with optional activation. \n
*
* Computes per-channel causal convolution across the sequence dimension:\n
* y[t, d] = Act( sum_{k=0}^{W-1} weight[k, d] * x[t-k, d] + bias[d] )\n
* where historical tokens x[-1], x[-2], ... come from the conv_states cache.\n
* Supports prefill (full sequence) and decode (incremental) modes.
*
* @par Inputs:
* @li x: Input sequence tensor. 2D varlen shape [cu_seq_len, dim] or 3D batch shape [batch, seqlen, dim].
* Supports float16, bfloat16.
 * @li weight: Convolution kernel filter. shape [K, dim] where K in {2, 3, 4}.
 * Weight layout: weight[0] applies to x[t], weight[1] to x[t-1], ..., weight[K-1] to x[t-K+1].
 * Same type as x.
 * @li conv_states: State cache tensor. shape [num_cache_lines, state_len, dim]. Stores historical tokens for
 * each cache line. Updated in-place by this operator. Same type as x.
 * state_len should be >= kernel_width - 1 for prefill, >= kernel_width for decode.
 * num_cache_lines must be >= batch, where batch is derived from x.shape[0] (3D batch mode) or
 * query_start_loc.size(0) - 1 (2D varlen mode).
 * @li bias: Optional. Convolution bias added after the weighted sum. shape [dim]. Same type as x.
* @li query_start_loc: Optional. Cumulative token offset for varlen input. shape [batch+1]. int32.
* Required when x is 2D; None when x is 3D batch.
* query_start_loc[0] must be 0, query_start_loc[-1] must equal cu_seq_len, values must be non-decreasing.
 * @li cache_indices: Optional. Maps each sequence to its cache slot in conv_states. shape [batch]. int32.
 * Value range [0, num_cache_lines). Sequences with cache_indices[i] == null_block_id are skipped.
 * Values other than null_block_id should not be duplicated, otherwise multiple sequences reading/writing
 * the same cache slot will cause history state errors and write conflicts.
 * When None, cache slot i is used for sequence i.
* @li initial_state_mode: Optional. Per-sequence flag: 1 = use cached conv_states as history,
* 0 = zero-initialize history (cold start). shape [batch]. int32. Value in {0, 1}.
* Only supported in prefill mode.
* @li num_accepted_tokens: Optional. Speculative decode support. Number of accepted tokens per sequence,
* used to compute the correct history offset within conv_states. shape [batch]. int32.
* Required in decode mode with speculative decoding; None otherwise.
* Only supported when kernel_width K = 4. Value in [0, state_len - (K - 1)].

* @par Attributes:
 * @li activation_mode: An optional string. Activation type applied after convolution: "silu" or "none". Default:
"silu".
* @li null_block_id: An optional int. Cache slot ID treated as padding (skipped). Default: 0 (no padding).
* Should be set to a value not present in cache_indices (e.g. -1) to avoid skipping valid sequences.
 * @par Outputs:
* @li conv_states: Updated state cache tensor. Last state_len tokens of (history || x) written back per sequence.
* Same shape and type as input conv_states.
* @li y: Output sequence tensor. In prefill mode, first (W-1) tokens are bias-only if initial_state_mode=0.
* Same shape and type as x.
*/
REG_OP(CausalConv1d)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .ATTR(activation_mode, String, "silu")
    .ATTR(null_block_id, Int, 0)
    .OUTPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CausalConv1d)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_
