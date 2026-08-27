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
 * \file fused_infer_attention_score_proto.h
 * \brief fused_infer_attention_score operator prototype definition
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_PROTO_H_
#define FUSED_INFER_ATTENTION_SCORE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Function FusedInferAttentionScore.
 *
 * @par Inputs:
 * @li query: A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn.
 * Query for attention structure.
 * @li key: It's a dynamic input. A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2,
 * float8_e4m3fn, int4, float4_e1m2, float4_e2m1. Key for attention structure.
 * @li value: It's a dynamic input. A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2,
 * float8_e4m3fn, int4, float4_e1m2, float4_e2m1. Value for attention structure.
 * @li pse_shift: A matrix tensor. The type support float16, bfloat16.
 * Postitional encoding parameters inside the attention structure.
 * @li atten_mask: A matrix tensor. The type support float16, bool, uint8, int8.
 * Mask of the result QK, indicating whether to calculate the correlation between tokens.
 * @li actual_seq_lengths: A matrix tensor. The type support int64.
 * Efective sequence length of query in different batches.
 * @li actual_seq_lengths_kv: A matrix tensor. The type support int64.
 * Effective sequence length of key/value in different batches.
 * @li dequant_scale1: A matrix tensor. The type support uint64, float32.
 * Dequantization factor behind BMM1.
 * @li quant_scale1: A matrix tensor. The type support float32.
 * Quantification factor in front of BMM2.
 * @li dequant_scale2: A matrix tensor. The type support uint64, float32.
 * Dequantization factor behind BMM2
 * @li quant_scale2: A matrix tensor. The type support float32, bfloat16.
 * Quantization factor of the output.
 * @li quant_offset2: A matrix tensor. The type support float32, bfloat16.
 * Qantization offset of the output.
 * @li antiquant_scale: A matrix tensor. The type support float16, bfloat16.
 * Antiquantization factor.
 * @li antiquant_offset: A matrix tensor. The type support float16, bfloat16.
 * Antiquantization offset.
 * @li block_table: A matrix tensor. The type support int32.
 * The block mapping table used in KV storage of PageAttention.
 * @li query_padding_size: A matrix tensor. The type support int64.
 * Whether the data in each batch of query is right-aligned, and the number of right-aligned entries.
 * @li kv_padding_size: A matrix tensor. The type support int64.
 * Whether the data in each batch of key/value is right-aligned, and the number of right-aligned entries.
 * @li key_antiquant_scale: A matrix tensor. The type support float16, bfloat16, float32.
 * The dequantization factor of key when separating the KV antiquantization parameters.
 * @li key_antiquant_offset: A matrix tensor. The type support float16, bfloat16, float32.
 * The dequantization offset of key when separating the KV antiquantization parameters.
 * @li value_antiquant_scale: A matrix tensor. The type support float16, bfloat16, float32.
 * The dequantization factor of value when separating the KV antiquantization parameters.
 * @li value_antiquant_offset: A matrix tensor. The type support float16, bfloat16, float32.
 * The dequantization offset of value when separating the KV antiquantization parameters.
 * @li key_shared_prefix: A matrix tensor. The type support int8, float16, bfloat16.
 * The input of the system prefix part of key in the attention structure.
 * @li value_shared_prefix: A matrix tensor. The type support int8, float16, bfloat16.
 * The input of the system prefix part of value in the attention structure.
 * @li actual_shared_prefix_len: A matrix tensor. The type support int64.
 * Effective Sequence Length of key_shared_prefix/value_shared_prefix.
 * @li query_rope: A matrix tensor. The type support int8, float16, bfloat16.
 * The rope information of query in the MLA structure.
 * @li key_rope: A matrix tensor. The type support int8, float16, bfloat16.
 * The rope information of key in the MLA structure.
 * @li key_rope_antiquant_scale: A matrix tensor. The type support float16, bfloat16.
 * The dequantization factor of the rope information of key in the MLA structure.
 * @li dequant_scale_query: A matrix tensor. The type support float32.
 * The dequantization factor of query.
 * @li learnable_sink: A matrix tensor. The type support bfloat16.
 * The sink token factor of attention score.
 * @li q_start_idx: A matrix tensor. The type support int64.
 * start idx of q for alibi pse.
 * @li kv_start_idx: A matrix tensor. The type support int64.
 * start idx of kv for alibi pse.
 *
 * @par Attributes:
 * @li num_heads: An int. The number of the heads.
 * @li scale: A float. The scale value. Default: 1.0.
 * @li pre_tokens: An int. Previous tokens. Default: 2147483647.
 * @li next_tokens: An int. Next tokens. Default: 2147483647.
 * @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND",
 * "BNSD_BSND"]. Default: "BSH".
 * @li num_key_value_heads: key value num heads. Default: 0.
 * @li sparse_mode: sparse mode. Default: 0.
 * - 0: default mask
 * - 1: all mask
 * - 2: leftUpCausal mask
 * - 3: rightDownCausal make
 * - 4: band mask
 * @li inner_precise: An int. 0, float16 high precision. 1, high performance. 2, high precision with row invalid fix.
 * 3, high performance with row invalid fix. Default: 1.
 * @li block_size: An int. Default: 0.
 * In PageAttention, KV stores the maximum number of tokens in each block.
 * @li antiquant_mode: An int. Antiquantization mode, 0: per-channel (per-channel includes per-tensor); 1: per-token.
 * Default: 0.
 * @li softmax_lse_flag: A bool. Whether to output softmax_lse. Default: false.
 * @li key_antiquant_mode: An int. Antiquantization mode of key. Default: 0.
 * - 0: per-channel (per-channel includes per-tensor)
 * - 1: per-token
 * - 2: per-tensor+per-head
 * - 3: per-token+per-head
 * - 4: per-token+using page attention mode to manage scale/offset
 * - 5: per-token+per-head+using page attention mode to manage scale/offset
 * - 6: per-token-group
 * @li value_antiquant_mode: An int. Antiquantization mode of key. The mode number is the same as
 * key_antiquant_mode. Default: 0.
 * @li query_quant_mode: An int. Quantization mode of key. Only support mode 3: per-token+per-head now. Default: 0.
 * @li pse_type: An int. Type of pse, 0: generate pse outside, 2: generate pse inside, 3: generate pse inside and do
 * sqrt. Default: 0.
 * @li out_dtype: An int. Dtype of attention_out tensor. Only supprt in PTA graph mode.
 * - 5: fp16, 15: bf16, 23: fp8 e5m2, 24: fp8 e4m3, 290: hifp8. Default: 0.
 *
 * @par Outputs:
 * @li attention_out: A matrix tensor. The type support float16, int8, bfloat16.
 * The output of attention structure.
 * @li softmax_lse: A matrix tensor. The type support float32.
 * Ring attention takes the result of multiplying query by key, first finds the max to get softmax_max.
 * The result of query multiplied by key minus softmax_max is then exponentiated, followed by summing to obtain
 * softmax_sum. Finally, take the log of softmax_sum and add the result obtained from softmax_max.
 *
 * @attention Constraints:
 * @li Ensure CANN and PyTorch package version compatibility when using this interface with PyTorch.
 * @li Handle empty input: Check if 'query' is empty; return if so.
 * If 'query' is non-empty and 'key', 'value' are empty tensors (i.e., S2=0), output a zero-filled tensor of the
 * corresponding shape (fill 'attention_out'). If 'attention_out' is an empty tensor, AscendCLNN framework will
 * handle it.
 * @li The shapes of tensors corresponding to 'key' and 'value' must be identical;
 * in non-continuous scenarios, the batch size in the tensor list of 'key' and 'value' must be 1, equal to the
 * number of 'query', with B, N, and D being equal.
 */
#ifndef OPS_PROTO_DEF_FUSEDINFERATTENTIONSCORE
#define OPS_PROTO_DEF_FUSEDINFERATTENTIONSCORE
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                    DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                      DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)
#endif

} // namespace ge

#endif // FUSED_INFER_ATTENTION_SCORE_PROTO_H_
