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
 * \file und_gen_qkv_rms_norm_rope_cache_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Fused QKV pre-processing for multimodal inference: concatenates the undecoded and generated QKV
* segments through cat_indices, applies RMSNorm and MRoPE on Q and K per token, outputs Q and scatters
* K/V into the paged KV cache according to slot_mapping. \n

* @par Inputs:
* Twelve inputs, including:
* @li und_qkv: A tensor, support BF16, shape [und_len, N, D], N = Hq + Hk + Hv.
* @li und_weights_q: A tensor, support BF16, shape [D], the RMSNorm weight of Q for the undecoded segment.
* @li und_weights_k: A tensor, support BF16, shape [D], the RMSNorm weight of K for the undecoded segment.
* @li cos_sin_cache: A tensor, support FP32, shape [max_pos, D], the first D/2 columns are cos and
* the last D/2 columns are sin.
* @li k_cache: A tensor, support BF16, shape [Bn, Bs, Hk, D], the paged KV cache of K, updated in place.
* @li v_cache: A tensor, support BF16, shape [Bn, Bs, Hv, D], the paged KV cache of V, updated in place.
* @li slot_mapping: A tensor, support INT64, shape [T], slot = block_num * Bs + row_idx.
* @li positions: A tensor, support INT64, shape [3, T], the time/height/width positions of MRoPE.
* @li gen_qkv: An optional tensor, support BF16, shape [gen_len, N, D].
* @li gen_weights_q: An optional tensor, support BF16, shape [D], the RMSNorm weight of Q for the
* generated segment.
* @li gen_weights_k: An optional tensor, support BF16, shape [D], the RMSNorm weight of K for the
* generated segment.
* @li cat_indices: An optional tensor, support INT64, shape [T], maps each output token to its source token. \n

* @par Attributes:
* @li num_heads_q: A required scalar, the head number of Q.
* @li num_heads_k: A required scalar, the head number of K.
* @li num_heads_v: A required scalar, the head number of V.
* @li norm_eps: An optional scalar, the epsilon of RMSNorm. Defaults to 1e-6.
* @li mrope_section: An optional scalar list of length 3, the three-axis section of MRoPE.
* Defaults to (), which degenerates to the standard RoPE. \n

* @par Outputs:
* Three outputs, including:
* @li q: A tensor, support BF16, shape [T, Hq, D], T = und_len + gen_len.
* @li k_cache: A tensor, support BF16, shape [Bn, Bs, Hk, D], shares the address with the input k_cache.
* @li v_cache: A tensor, support BF16, shape [Bn, Bs, Hv, D], shares the address with the input v_cache. \n

* @attention Constraints:
* @li D is fixed to 128, and (num_heads_q, num_heads_k, num_heads_v) supports (8,1,1) and (16,2,2) only.
* @li k_cache and v_cache must be contiguous, and their capacity must satisfy Bn * Bs >= T.
* @li The four optional inputs must be either all provided or all absent within one call. \n
*/
REG_OP(UndGenQkvRmsNormRopeCache)
    .INPUT(und_qkv, TensorType({DT_BF16}))
    .INPUT(und_weights_q, TensorType({DT_BF16}))
    .INPUT(und_weights_k, TensorType({DT_BF16}))
    .INPUT(cos_sin_cache, TensorType({DT_FLOAT}))
    .INPUT(k_cache, TensorType({DT_BF16}))
    .INPUT(v_cache, TensorType({DT_BF16}))
    .INPUT(slot_mapping, TensorType({DT_INT64}))
    .INPUT(positions, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(gen_qkv, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(gen_weights_q, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(gen_weights_k, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(cat_indices, TensorType({DT_INT64}))
    .OUTPUT(q, TensorType({DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_BF16}))
    .OUTPUT(v_cache, TensorType({DT_BF16}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_k, Int)
    .REQUIRED_ATTR(num_heads_v, Int)
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(mrope_section, ListInt, {})
    .OP_END_FACTORY_REG(UndGenQkvRmsNormRopeCache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
