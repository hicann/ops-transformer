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
 * \file fused_causal_conv1d_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSED_CAUSAL_CONV1D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSED_CAUSAL_CONV1D_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Function FusedCausalConv1d.
 *
 * @par Inputs:
 * @li x: Input sequence tensor. shape [cu_seq_len, dim] or [batch, seqlen, dim]. Supports float16, bfloat16.
 * @li weight: Convolution kernel of shape [K, dim], K fixed to 3. Same type as x.
 * @li conv_states: Cache state tensor storing historical tokens per sequence, updated in-place. Same type as x.
 * @li query_start_loc: Optional. Start offset of each sequence in x. shape [batch+1]. int32.
 * @li cache_indices: Optional. Index mapping each sequence to its cache slot. 1D [batch] or 2D [batch, max_num_blocks].
 * int32.
 * @li initial_state_mode: Optional. Flag indicating whether each sequence uses cached data. shape [batch]. int32.
 * @li bias: Optional. Convolution bias of shape [dim]. Same type as x.
 * @li num_accepted_tokens: Optional. Number of accepted tokens per sequence in speculative decoding. shape [batch].
 * int32.
 * @li num_computed_tokens: Optional. Total computed tokens per batch for initial state detection. shape [batch]. int32.
 * @li block_idx_first_scheduled_token: Optional. Block index of first scheduled token per batch (APC). shape [batch].
 * int32.
 * @li block_idx_last_scheduled_token: Optional. Block index of last scheduled token per batch (APC). shape [batch].
 * int32.
 * @li initial_state_idx: Optional. Initial state block index per batch (APC). shape [batch]. int32.
 *
 * @par Attributes:
 * @li activation_mode: A optional int. Activation function type: 0 (None), 1 (silu), 2 (swish).
 * @li pad_slot_id: A optional int. Slot ID used to skip padding batches.
 * @li run_mode: A optional int. Execution mode (legacy, not supported).
 * @li max_query_len: A optional int. Max sequence length across all batches.
 * @li residual_connection: A optional int. Whether to use residual connection: 0 (no), 1 (yes).
 * @li block_size: A optional int. APC block size, supports 128/256.
 * @li conv_mode: A optional int. Convolution mode: 0 (Qwen3-Next), 1 (Pangu v2).
 * @li max_draft_tokens: A optional int. Maximum speculative tokens, supports 0~16.
 *
 * @par Outputs:
 * @li conv_states: Updated cache state tensor. Same shape and type as input conv_states.
 * @li y: Output sequence tensor. Same shape and type as x.
 */
REG_OP(FusedCausalConv1d)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(num_computed_tokens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_idx_first_scheduled_token, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_idx_last_scheduled_token, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_state_idx, TensorType({DT_INT32}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .ATTR(run_mode, Int, 0)
    .ATTR(max_query_len, Int, -1)
    .ATTR(residual_connection, Int, 1)
    .ATTR(block_size, Int, 128)
    .ATTR(conv_mode, Int, 0)
    .ATTR(max_draft_tokens, Int, 7)
    .OUTPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(FusedCausalConv1d)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_FUSED_CAUSAL_CONV1D_H_
