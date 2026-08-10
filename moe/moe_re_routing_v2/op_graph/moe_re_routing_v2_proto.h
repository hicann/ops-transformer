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
 * \file moe_re_routing_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_RE_ROUTING_V2_OPS_H_
#define OPS_OP_PROTO_INC_MOE_RE_ROUTING_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Rearrange tokens from rank order to expert order, and optionally rearrange topk weight
* @par Inputs:
* @li tokens: A 2D tensor, represents tokens in rank-order. Type is BFloat16, Float16, DT_FLOAT8_E5M2,
      DT_FLOAT8_E4M3FN, Int8, DT_HIFLOAT8, DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2. Shape supports (A, H). Format supports ND.
* @li expert_token_num_per_rank: A 2D tensor, represents numbers of tokens belong to an expert on specific rank.
      Type is Int32 or Int64. Shape supports (N, E). Format supports ND.
* @li per_token_scales: A 1D, 2D or 3D tensor, optional, represents tokens scale in rank-order.
      Type is Float32, DT_FLOAT8_E8M0.
      Shape supports (A), (A,S) or (A,K/64,2). Format supports ND. If tokens is FLOAT8,
      per_token_scales must be DT_FLOAT8_E8M0. \n
  Ascend 950PR/Ascend 950DT support 1D, 2D or 3D (3D only for FP8 tokens). \n
* @li expert_topk_weight: A 2D tensor, optional, represents topk weight of each token in rank-order.
      Type is Float32. Shape supports (A, 1). Format supports ND. Must be used together with
      permute_topk_weight: either both provided or neither. \n
* @par Outputs:
* @li permute_tokens: A 2D tensor, represents tokens in expert-order. Type is BFloat16, Float16,
      Int8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2. Shape supports (A, H).
  Format supports ND.
* @li permute_per_token_scales: A 1D, 2D or 3D tensor, represents tokens scale in expert-order.
      Type is Float32, DT_FLOAT8_E8M0. Shape supports (A), (A,S) or (A,K/64,2). Format supports ND.
  Ascend 950PR/Ascend 950DT support 1D, 2D or 3D (3D only for FP8 tokens). \n
* @li permute_token_idx: A 1D tensor, represents token idx in rank-order. Type is Int32.
      Shape supports (A). Format supports ND.
* @li expert_token_num: A 1D tensor, represents tokens nums of experts. Type is Int32 or Int64.
      Shape supports (E). Format supports ND.
* @li permute_topk_weight: A 2D tensor, optional, represents topk weight in expert-order, corresponds
      one-to-one with permute_tokens. Type is Float32. Shape supports (A, 1). Format supports ND.
      Must be used together with expert_topk_weight: either both provided or neither. \n
* @par Attributes:
* @li expert_token_num_type: Optional integer, represents the cumsum or count mode. Type is Int. Default: 1. Value
      supports 0-cumsum or 1-count, currently only supports "1".
* @li idx_type: Optional integer, represents the gather or scatter index. Type is Int. Default: 0. Value
      supports 0-gather idx or 1-scatter idx. Ascend 950PR/Ascend 950DT supports "0" or "1".
*/
REG_OP(MoeReRoutingV2)
    .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                               DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(expert_topk_weight, TensorType({DT_FLOAT}))
    .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                                        DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(permute_token_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(permute_topk_weight, TensorType({DT_FLOAT}))
    .ATTR(expert_token_num_type, Int, 1)
    .ATTR(idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeReRoutingV2)

} // namespace ge

#endif
